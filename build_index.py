"""构建/重建岗位向量库：读 jobs.json -> 硅基流动 bge-m3 嵌入 -> Chroma 持久化。"""
import json
import chromadb
from openai import OpenAI

API_KEY = "sk-kxicoembqloskhdqerfuzpkcrstrfiozbfmlokutxikrqdqz"
EMBED_MODEL = "BAAI/bge-m3"
CHROMA_DIR = "./chroma_db"


def make_client():
    return OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")


def embed(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    data = sorted(resp.data, key=lambda x: x.index)
    return [d.embedding for d in data]


def job_to_text(j):
    return (
        f"{j['company']} | {j['title']} | 城市:{j['city']} | 类型:{j['type']} | "
        f"薪资:{j['salary']} | 学历:{j['edu']} | 经验:{j['exp']} | "
        f"要求:{j['requirements']} | 标签:{','.join(j.get('tags', []))}"
    )


def main():
    jobs = json.load(open("jobs.json", encoding="utf-8"))
    client = make_client()
    cdb = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        cdb.delete_collection("jobs")
    except Exception:
        pass
    col = cdb.create_collection(name="jobs")

    texts = [job_to_text(j) for j in jobs]
    metas = [
        {
            "company": j["company"], "title": j["title"], "city": j["city"],
            "salary": j["salary"], "type": j["type"],
            "tags": ",".join(j.get("tags", [])),
        }
        for j in jobs
    ]
    ids = [str(j["id"]) for j in jobs]

    B = 32
    embs = []
    for i in range(0, len(texts), B):
        embs += embed(client, texts[i:i + B])
        print(f"  嵌入 {min(i+B, len(texts))}/{len(texts)}")

    col.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)
    print(f"索引完成：共 {col.count()} 条岗位")


if __name__ == "__main__":
    main()
