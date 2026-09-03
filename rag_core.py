"""RAG 共享核心：配置、客户端、嵌入、岗位文本化、检索（混合重排）、LLM 对话与流式生成。
app.py 与 build_index.py、collect_jobs.py 共用本模块，避免重复代码。

本版本优化点（对比初版）：
  - 客户端带 timeout + max_retries，抗 429/超时更稳。
  - 查询嵌入结果 LRU 缓存，重复/相似问题秒回。
  - 混合检索：向量相似 + 中文 bigram 关键词重叠，RRF 式融合重排，top-k 召回更准。
  - 可选 bge-reranker 交叉编码器重排：向量召回 top-20 候选再精排 top-5，精度更高（v5）。
  - rewrite_query 带重试；生成失败有降级兜底答案。
"""
import os
import re
import json
import time
import threading
import urllib.request
from openai import OpenAI

# ---------- 配置 ----------
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # 交叉编码器重排，提升召回精度
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


PLACEHOLDER_KEY = "在此填写你的key"


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


def ensure_env_file():
    """冷启动友好：若项目根目录没有 .env，则从 .env.example 复制一份，
    避免新手 clone 后因缺 .env 而启动即空库。返回 True 表示本次新建了 .env。"""
    import shutil
    root = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(root, ".env")
    ex_path = os.path.join(root, ".env.example")
    if not os.path.exists(env_path) and os.path.exists(ex_path):
        shutil.copyfile(ex_path, env_path)
        return True
    return False


# 冷启动：先确保 .env 存在（新手 clone 后常缺），再读入环境变量
ensure_env_file()
_load_dotenv()

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")


def is_key_ready():
    """密钥是否有效：非空且不是 .env.example 里的占位符。"""
    return bool(API_KEY) and API_KEY != PLACEHOLDER_KEY


def preflight():
    """启动前自检：密钥未就绪时打印醒目指引并返回 False。"""
    if is_key_ready():
        return True
    print("\n" + "=" * 64)
    print("⚠️  尚未配置有效的 SILICONFLOW_API_KEY，服务将无法正常回答。")
    print("-" * 64)
    print("  请按以下步骤操作（只需一次）：")
    print("    1) 用编辑器打开项目根目录的 .env 文件")
    print("    2) 把  SILICONFLOW_API_KEY=在此填写你的key")
    print("       改成你的真实 key，例如  SILICONFLOW_API_KEY=sk-xxxx")
    print("    3) 保存文件，重新运行启动命令")
    print("  免费获取 key：https://cloud.siliconflow.cn")
    print("=" * 64 + "\n")
    return False


def make_client():
    if not is_key_ready():
        raise RuntimeError(
            "未配置有效的 SILICONFLOW_API_KEY。请打开项目根目录的 .env，"
            "把 SILICONFLOW_API_KEY 改成你的硅基流动 key（https://cloud.siliconflow.cn 免费获取），"
            "保存后重新运行。"
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


# ---------- 职能类别推断（关键词，用于 /add 与采集脚本补全 category） ----------
CATEGORY_KEYWORDS = {
    "技术研发": ["算法", "开发", "程序", "软件", "前端", "后端", "全栈", "测试", "运维",
              "数据", "ai", "人工智能", "机器学习", "深度学习", "自然语言", "大数据",
              "嵌入式", "硬件", "网络", "安全", "架构", "技术", "研发", "python", "java", "c++"],
    "产品": ["产品", "pm", "需求", "prd", "产品经理"],
    "设计": ["ui", "ux", "视觉", "设计", "交互", "原型", "美工", "平面"],
    "运营": ["运营", "新媒体", "社群", "活动运营", "内容运营", "用户运营", "短视频运营"],
    "市场/营销": ["市场", "营销", "品牌", "商务拓展", "公关", "渠道", "bd", "市场策划"],
    "销售": ["销售", "大客户", "售前", "电话销售", "房产销售", "汽车销售", "导购"],
    "客服/客户成功": ["客服", "客户成功", "售后", "工单", "会员运营"],
    "人力资源(HR)": ["招聘", "人力资源", "薪酬", "绩效", "hrbp", "hr", "培训", "员工关系"],
    "行政/文秘": ["行政", "前台", "文秘", "会务", "档案", "总助", "助理"],
    "财务/会计": ["会计", "财务", "出纳", "审计", "税务", "成本"],
    "法务/合规": ["法务", "合规", "律师", "合同", "知识产权", "风控合规"],
    "供应链/物流": ["采购", "供应链", "物流", "仓储", "关务", "库存", "运输", "计划专员"],
    "生产/制造": ["工艺", "设备", "质量", "生产", "制造", "qc", "焊接", "装配", "精益", "注塑", "质检"],
    "医疗/健康": ["护士", "药剂", "临床", "医疗", "健康", "医学", "检验", "康复", "营养", "crc"],
    "教育培训": ["教师", "助教", "课程", "教育", "培训", "留学", "早教", "教研", "辅导"],
    "金融/银行": ["银行", "金融", "风控", "投资", "保险", "证券", "信贷", "理财", "资产"],
    "房地产/建筑/工程": ["土木", "造价", "施工", "房产", "建筑", "结构", "bim", "规划", "工程"],
    "媒体/内容": ["编辑", "记者", "文案", "短视频", "直播", "摄影", "内容", "媒体", "编导"],
    "翻译/语言": ["翻译", "英语", "日语", "小语种", "本地化", "口译", "笔译"],
    "餐饮/酒店/零售": ["餐饮", "酒店", "零售", "厨师", "服务员", "店长", "烘焙", "咖啡", "门店"],
    "综合/管培": ["管培", "储备干部", "项目助理", "战略", "商业分析", "轮岗"],
}


def classify_category(title="", requirements="", default="综合/管培"):
    """根据岗位名 + 要求文本，关键词推断职能类别。

    - 中文关键词按子串匹配（中文无词边界）。
    - 英文/数字关键词按「独立词」匹配（正则词边界），避免 ai/it/ui 等误命中
      retail、training、audit 等含相同字母的英文词。
    - 遍历顺序即优先级：更具体的类目放前面，避免被宽泛类目抢先命中。
    """
    import re
    text = f"{(title or '')} {(requirements or '')}".lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw.isascii():
                if re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", text):
                    return cat
            elif kw in text:
                return cat
    return default


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
        f"{j['company']} | {j['title']} | 职能:{j.get('category', '')} | 城市:{j['city']} | 类型:{j['type']} | "
        f"薪资:{j['salary']} | 学历:{j['edu']} | 经验:{j['exp']} | "
        f"要求:{j['requirements']} | 标签:{','.join(j.get('tags', []))}"
    )


def job_meta(j):
    return {
        "company": j["company"], "title": j["title"],
        "category": j.get("category", ""),
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
    """把 {city,type,edu,category} 转为 Chroma where 子句(AND)。无过滤返回 None。纯函数，便于单测。"""
    if not filters:
        return None
    conds = []
    if filters.get("city"):
        conds.append({"city": filters["city"]})
    if filters.get("type"):
        conds.append({"type": filters["type"]})
    if filters.get("edu"):
        conds.append({"edu_norm": filters["edu"]})
    if filters.get("category"):
        conds.append({"category": filters["category"]})
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


# ---------- 交叉编码器重排(bge-reranker) ----------
def rerank_docs(query, docs, top_n=5, model=RERANK_MODEL, retries=3):
    """用硅基流动 bge-reranker 对候选文档按与 query 的相关性重排，返回重排后的下标列表(降序)。
    失败(无 key / 网络 / 限流)返回 None，由调用方降级为混合重排。"""
    if not API_KEY or not docs:
        return None
    url = BASE_URL + "/rerank"
    body = json.dumps({
        "model": model,
        "query": query,
        "documents": docs,
        "top_n": min(top_n, len(docs)),
        "return_documents": False,
    }).encode("utf-8")
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=CLIENT_TIMEOUT) as r:
                data = json.load(r)
            results = data.get("results", [])
            if not results:
                return None
            return [x["index"] for x in results]
        except Exception as e:
            last = e
            time.sleep(1.0)
    return None


def rerank_scores(query, docs, model=RERANK_MODEL, retries=3):
    """返回 {doc_index: relevance_score} 映射，作为「相关性裁判」供评估使用（不改动检索本身）。"""
    if not API_KEY or not docs:
        return {}
    url = BASE_URL + "/rerank"
    body = json.dumps({
        "model": model,
        "query": query,
        "documents": docs,
        "top_n": len(docs),
        "return_documents": False,
    }).encode("utf-8")
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=CLIENT_TIMEOUT) as r:
                data = json.load(r)
            results = data.get("results", [])
            return {x["index"]: x["relevance_score"] for x in results}
        except Exception as e:
            last = e
            time.sleep(1.0)
    return {}


def retrieve(client, col, q, n=5, k=20, filters=None, rerank=True):
    """向量召回 k 条候选，可选经 bge-reranker 交叉编码器重排取 top-n。返回 (documents, metadatas)。
    - rerank=True（默认）：向量 top-k 候选 → 交叉编码器重排 → top-n，精度更高。
    - rerank=False：保持原「向量 + 中文 bigram RRF」混合重排（评估基线用）。
    - reranker 调用失败时自动降级为混合重排，保证「永远有结果」。
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
    if rerank:
        order = rerank_docs(q, docs, top_n=n)
        if order is not None:
            return [docs[i] for i in order], [metas[i] for i in order]
        # 重排失败 → 降级
    return _hybrid_rerank(q, docs, metas, dists, n)


# ---------- LLM：查询改写(智能检索) ----------
_rewrite_cache = {}   # (问题, 模型) -> 改写结果；相同问题免再调一次 LLM，降延迟

def rewrite_query(client, q, model=LLM_MODEL):
    """把用户口语/模糊问题改写成更适合向量检索的关键词短语(中文,<=40字)。带重试。
    相同问题(同模型)命中缓存直接返回，省一次 LLM 调用、降延迟。"""
    key = (q, model)
    if key in _rewrite_cache:
        return _rewrite_cache[key]
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
            result = out or q
            _rewrite_cache[key] = result
            return result
        except Exception as e:
            last = e
            time.sleep(1.0)
    _rewrite_cache[key] = q
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
