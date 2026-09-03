"""RAG 共享核心：配置、客户端、嵌入、岗位文本化、检索（混合重排）、LLM 对话与流式生成。
app.py 与 build_index.py、collect_jobs.py 共用本模块，避免重复代码。

本版本优化点（对比初版）：
  - 客户端带 timeout + max_retries，抗 429/超时更稳。
  - 查询嵌入结果 LRU 缓存，重复/相似问题秒回。
  - 混合检索：向量相似 + 中文 bigram 关键词重叠，RRF 式融合重排，top-k 召回更准。
  - rewrite_query 带重试；生成失败有降级兜底答案。
"""
import os
import re
import json
import time
import threading
from openai import OpenAI

# ---------- 配置 ----------
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3")      # 默认：标准快模型
LLM_MODEL_REASON = "deepseek-ai/DeepSeek-R1"                         # 深度思考（推理）模型
CHROMA_DIR = "./chroma_db"
BASE_URL = "https://api.siliconflow.cn/v1"
COLLECTION = "jobs"
CLIENT_TIMEOUT = 60          # 单请求超时（秒）
CLIENT_MAX_RETRIES = 3       # SDK 自带退避重试

# 检索/统计归一化用的核心城市
CITY_BASE = ["无锡", "苏州", "南京", "上海", "常州", "南通", "杭州", "宁波"]

SYSTEM_PROMPT = (
    "你是一名面向大学生的实习/校招/社招岗位智能咨询助手，专注长三角"
    "(无锡、苏州、南京、上海、常州、南通、杭州、宁波)技术类岗位。\n"
    "你会拿到若干条【检索到的真实岗位资料】，请基于这些资料回答，并可结合一般求职常识给出建议。\n"
    "要求：\n"
    "1) 优先使用资料中的真实信息(公司、岗位、城市、薪资、学历、经验、要求)，并在关键结论处用【公司·岗位】标注来源；\n"
    "2) 当资料不足以回答时，如实说明“资料中未收录”，不要编造公司名与薪资；\n"
    "3) 回答结构化、分点，必要时做岗位对比或人岗匹配度分析；\n"
    "4) 给出可行动建议(如投递优先级、技能补足方向)。\n"
    "若用户是在多轮对话中追问，请结合上下文连贯回答。"
)


def _load_dotenv():
    """从项目根目录 .env 读取环境变量(不依赖第三方包)。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")


def make_client():
    if not API_KEY:
        raise RuntimeError(
            "未找到 SILICONFLOW_API_KEY。请复制 .env.example 为 .env 并填入你的硅基流动 key，"
            "或将环境变量 SILICONFLOW_API_KEY 导出后再运行。"
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL,
                  timeout=CLIENT_TIMEOUT, max_retries=CLIENT_MAX_RETRIES)


# ---------- 城市归一化(看板/统计用) ----------
def normalize_city(c: str) -> str:
    """优先取字符串中出现的首个核心城市(按 CITY_BASE 顺序)，覆盖复合/噪声写法。
    例：'无锡新吴'->'无锡'、'全国(含上海)'->'上海'、'北京/上海'->'上海'、
        '南京/无锡/成都'->'无锡'(取 CITY_BASE 中首个命中)。
        完全不含核心城市则返回原值(统计时再归并「其他」)。"""
    for base in CITY_BASE:
        if base in c:
            return base
    if "/" in c:
        for part in c.split("/"):
            for base in CITY_BASE:
                if base in part:
                    return base
    return c


def normalize_edu(e):
    """把杂乱的学历字段归并为看板用的少数桶（与 app.py 共用，统一口径）。"""
    e = (e or "").replace("（", "(").replace("）", ")")
    if "博士" in e:
        return "博士"
    if "硕士" in e:
        return "硕士"
    if "大专" in e or "中专" in e:
        return "大专/中专"
    if "不限" in e:
        return "不限"
    if "本科" in e:
        return "本科"
    return "其他"


# ---------- 中文 bigram(用于关键词检索信号) ----------
def _bigrams(s: str):
    s = re.sub(r"\s+", "", s or "")
    return set(s[i:i + 2] for i in range(len(s) - 1))


# ---------- 嵌入(带重试 + 查询缓存) ----------
_embed_cache = {}          # 查询文本 -> 向量，进程内复用
_embed_cache_lock = threading.Lock()
EMBED_CACHE_MAX = 2000


def embed_texts(client, texts):
    last = None
    for _ in range(3):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            data = sorted(resp.data, key=lambda x: x.index)
            return [d.embedding for d in data]
        except Exception as e:
            last = e
            time.sleep(1.5)
    raise last or RuntimeError("embedding failed")


def embed_query(client, q):
    with _embed_cache_lock:
        if q in _embed_cache:
            return _embed_cache[q]
    emb = embed_texts(client, [q])[0]
    with _embed_cache_lock:
        if len(_embed_cache) < EMBED_CACHE_MAX:
            _embed_cache[q] = emb
    return emb


# ---------- 岗位 <-> 文本 ----------
def job_to_text(j):
    return (
        f"{j['company']} | {j['title']} | 城市:{j['city']} | 类型:{j['type']} | "
        f"薪资:{j['salary']} | 学历:{j['edu']} | 经验:{j['exp']} | "
        f"要求:{j['requirements']} | 标签:{','.join(j.get('tags', []))}"
    )


def job_meta(j):
    return {
        "company": j["company"], "title": j["title"],
        "city": normalize_city(j["city"]), "salary": j["salary"],
        "type": j["type"], "tags": ",".join(j.get("tags", [])),
        "edu": j.get("edu", ""), "edu_norm": normalize_edu(j.get("edu", "")),
        "exp": j.get("exp", ""), "source": j.get("source", ""),
    }


def load_jobs(path="jobs.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_collection(client=None, path="jobs.json",
                     chroma_dir=CHROMA_DIR, batch=64, verbose=True):
    """读 jobs.json -> 硅基流动 bge-m3 嵌入 -> Chroma 持久化。返回 collection。"""
    import chromadb
    if client is None:
        client = make_client()
    jobs = load_jobs(path)
    cdb = chromadb.PersistentClient(path=chroma_dir)
    # 复用同名 collection：清空旧数据后追加，避免「删除+重建」让已持有的句柄失效
    try:
        col = cdb.get_collection(COLLECTION)
        existing = col.get(include=[])
        if existing and existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        col = cdb.create_collection(name=COLLECTION)

    texts = [job_to_text(j) for j in jobs]
    metas = [job_meta(j) for j in jobs]
    ids = [str(j["id"]) for j in jobs]

    embs = []
    for i in range(0, len(texts), batch):
        embs += embed_texts(client, texts[i:i + batch])
        if verbose:
            print(f"  嵌入 {min(i + batch, len(texts))}/{len(texts)}")
    col.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)
    if verbose:
        print(f"索引完成：共 {col.count()} 条岗位")
    return col


# ---------- 混合检索 + 重排 ----------
def build_where(filters):
    """把 {city,type,edu} 转为 Chroma where 子句(AND)。无过滤返回 None。纯函数，便于单测。"""
    if not filters:
        return None
    conds = []
    if filters.get("city"):
        conds.append({"city": filters["city"]})
    if filters.get("type"):
        conds.append({"type": filters["type"]})
    if filters.get("edu"):
        conds.append({"edu_norm": filters["edu"]})
    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


def _hybrid_rerank(q, docs, metas, dists, n):
    """向量相似(1-dist) 与 中文bigram关键词重叠 融合，取 top-n。"""
    qbg = _bigrams(q)
    scored = []
    for d, m, dist in zip(docs, metas, dists):
        vec = 1.0 - dist
        dbg = _bigrams(d)
        overlap = len(qbg & dbg)
        kw = overlap / (len(qbg) + 1)
        score = 0.78 * vec + 0.22 * kw
        scored.append((score, d, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:n]], [x[2] for x in scored[:n]]


def retrieve(client, col, q, n=5, k=12, filters=None):
    """向量召回 k 条，再用关键词信号重排返回 top-n。返回 (documents, metadatas)。
    filters: {city,type,edu} 经 build_where 转为 Chroma where，实现按城市/类型/学历硬筛选。"""
    emb = embed_query(client, q)
    where = build_where(filters)
    res = col.query(query_embeddings=[emb], n_results=min(k, col.count() or 1),
                    where=where if where else None)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    if not docs:
        return [], []
    return _hybrid_rerank(q, docs, metas, dists, n)


# ---------- LLM：查询改写(智能检索) ----------
def rewrite_query(client, q, model=LLM_MODEL):
    """把用户口语/模糊问题改写成更适合向量检索的关键词短语(中文,<=40字)。带重试。"""
    sys_p = (
        "你是检索增强系统的查询改写器。请把用户的问题改写成一段更适合向量检索的"
        "中文关键词/短语，保留城市、岗位、技能、学历等关键信息，不超过40字，"
        "只输出改写结果，不要解释。"
    )
    last = None
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": q}],
                temperature=0.2,
            )
            out = (r.choices[0].message.content or "").strip()
            return out or q
        except Exception as e:
            last = e
            time.sleep(1.0)
    return q


# ---------- LLM：组织多轮消息 ----------
def build_messages(history, context, q):
    """history: list[{"role","content"}]; 返回完整 messages 列表。"""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "【检索到的岗位资料】\n" + context},
    ]
    for h in history[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": q})
    return msgs


# ---------- LLM：流式生成(带超时重试) ----------
def stream_answer(client, model, messages):
    """yield 文本片段(token)。失败抛异常由调用方处理。"""
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.3, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def fallback_answer(metas):
    """生成失败时的降级答案：直接列出检索到的岗位，保证“永远有结果”。"""
    lines = ["（生成模型暂时不可用，以下为检索到的相关岗位，请参考）\n"]
    for i, m in enumerate(metas, 1):
        lines.append(
            f"{i}. 【{m.get('company','')}·{m.get('title','')}】"
            f"{m.get('city','')} | {m.get('type','')} | {m.get('salary','')}"
        )
    return "\n".join(lines)


def resolve_model(alias):
    """把前端传入的 v3/r1/标准/深度思考 或完整模型名映射为实际模型 id。"""
    if not alias:
        return LLM_MODEL
    a = str(alias).lower()
    if a in ("r1", "深度思考", "reason", "reasoning"):
        return LLM_MODEL_REASON
    if a in ("v3", "标准", "fast", "default"):
        return LLM_MODEL
    return alias  # 允许直接传完整模型名
