"""RAG 共享核心：配置、客户端、嵌入、岗位文本化、索引构建。
app.py 与 build_index.py 共用本模块，避免重复代码。
"""
import os
import json
from openai import OpenAI

# ---------- 配置 ----------
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"
CHROMA_DIR = "./chroma_db"
BASE_URL = "https://api.siliconflow.cn/v1"
COLLECTION = "jobs"


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
        "company": j["company"], "title": j["title"], "city": j["city"],
        "salary": j["salary"], "type": j["type"],
        "tags": ",".join(j.get("tags", [])),
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
