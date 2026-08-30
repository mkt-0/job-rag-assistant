"""RAG 共享核心：配置、客户端、嵌入、岗位文本化、检索、LLM 对话与流式生成。
app.py 与 build_index.py 共用本模块，避免重复代码。
"""
import os
import json
from openai import OpenAI

# ---------- 配置 ----------
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3")      # 默认：标准快模型
LLM_MODEL_REASON = "deepseek-ai/DeepSeek-R1"                         # 深度思考（推理）模型
CHROMA_DIR = "./chroma_db"
BASE_URL = "https://api.siliconflow.cn/v1"
COLLECTION = "jobs"

# 检索/统计归一化用的核心城市
CITY_BASE = ["无锡", "苏州", "南京", "上海", "常州", "南通", "杭州", "宁波"]

SYSTEM_PROMPT = (
    "你是一名面向大学生的实习/校招/社招岗位智能咨询助手，专注长三角"
    "（无锡、苏州、南京、上海、常州、南通、杭州、宁波）技术类岗位。\n"
    "你会拿到若干条【检索到的真实岗位资料】，请基于这些资料回答，并可结合一般求职常识给出建议。\n"
    "要求：\n"
    "1) 优先使用资料中的真实信息（公司、岗位、城市、薪资、学历、经验、要求）；\n"
    "2) 当资料不足以回答时，如实说明“资料中未收录”，不要编造公司名与薪资；\n"
    "3) 回答结构化、分点，必要时做岗位对比或人岗匹配度分析；\n"
    "4) 给出可行动建议（如投递优先级、技能补足方向）。\n"
    "若用户是在多轮对话中追问，请结合上下文连贯回答。"
)


def _load_dotenv():
    """从项目根目录 .env 读取环境变量（不依赖第三方包）。"""
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
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ---------- 城市归一化（看板/统计用） ----------
def normalize_city(c: str) -> str:
    for base in CITY_BASE:
        if c.startswith(base):
            return base
    if "/" in c:
        return normalize_city(c.split("/")[0])
    return c


# ---------- 嵌入 ----------
def embed_texts(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    data = sorted(resp.data, key=lambda x: x.index)
    return [d.embedding for d in data]


def embed_query(client, q):
    return embed_texts(client, [q])[0]


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
    }


def load_jobs(path="jobs.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_collection(client=None, path="jobs.json",
                     chroma_dir=CHROMA_DIR, batch=32, verbose=True):
    """读 jobs.json -> 硅基流动 bge-m3 嵌入 -> Chroma 持久化。返回 collection。"""
    import chromadb
    if client is None:
        client = make_client()
    jobs = load_jobs(path)
    cdb = chromadb.PersistentClient(path=chroma_dir)
    try:
        cdb.delete_collection(COLLECTION)
    except Exception:
        pass
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


# ---------- 检索 ----------
def retrieve(client, col, q, n=5):
    """返回 (documents, metadatas) 各 n 条。"""
    emb = embed_query(client, q)
    res = col.query(query_embeddings=[emb], n_results=n)
    return res["documents"][0], res["metadatas"][0]


# ---------- LLM：查询改写（智能检索） ----------
def rewrite_query(client, q, model=LLM_MODEL):
    """把用户口语/模糊问题改写成更适合向量检索的关键词短语（中文，<=40字）。"""
    sys_p = (
        "你是检索增强系统的查询改写器。请把用户的问题改写成一段更适合向量检索的"
        "中文关键词/短语，保留城市、岗位、技能、学历等关键信息，不超过40字，"
        "只输出改写结果，不要解释。"
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_p},
                      {"role": "user", "content": q}],
            temperature=0.2,
        )
        out = r.choices[0].message.content.strip()
        return out or q
    except Exception:
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


# ---------- LLM：流式生成 ----------
def stream_answer(client, model, messages):
    """yield 文本片段（token）。"""
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.3, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


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
