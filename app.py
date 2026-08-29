"""岗位 RAG 交互服务：Flask + Chroma(硅基流动 bge-m3) + 硅基流动 LLM。
路由：
  GET  /        交互界面
  POST /ask     自然语言问答 -> {answer, sources}
  GET  /jobs    当前库岗位列表与数量
  POST /add     自助补充一条岗位（粘 JD 入库）
"""
import os
import json
import chromadb
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI


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
if not API_KEY:
    raise RuntimeError(
        "未找到 SILICONFLOW_API_KEY。请复制 .env.example 为 .env 并填入你的硅基流动 key，"
        "或将环境变量 SILICONFLOW_API_KEY 导出后再运行 app.py。"
    )
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"
CHROMA_DIR = "./chroma_db"

client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
app = Flask(__name__, static_folder=".", template_folder=".")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
col = chroma.get_or_create_collection(name="jobs")


def embed_query(q):
    r = client.embeddings.create(model=EMBED_MODEL, input=[q])
    return r.data[0].embedding


def job_to_text(j):
    return (
        f"{j['company']} | {j['title']} | 城市:{j['city']} | 类型:{j['type']} | "
        f"薪资:{j['salary']} | 学历:{j['edu']} | 经验:{j['exp']} | "
        f"要求:{j['requirements']} | 标签:{','.join(j.get('tags', []))}"
    )


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    q = (request.json or {}).get("question", "").strip()
    if not q:
        return jsonify({"error": "问题为空"}), 400
    q_emb = embed_query(q)
    res = col.query(query_embeddings=[q_emb], n_results=5)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    context = "\n\n".join(docs)
    prompt = (
        "你是一个面向大学生的实习/校招岗位咨询助手。只依据下方【资料】回答用户问题，"
        "资料来自真实招聘信息检索结果。若资料中无相关信息，请如实说明“资料中未收录”。"
        "回答要简洁、结构化，尽量给出可行动建议，并点出匹配/不匹配的原因。\n\n"
        f"【资料】\n{context}\n\n【用户问题】\n{q}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content
    sources = [
        {
            "company": m["company"], "title": m["title"], "city": m["city"],
            "salary": m["salary"], "type": m["type"], "tags": m.get("tags", ""),
        }
        for m in metas
    ]
    return jsonify({"answer": answer, "sources": sources})


@app.route("/jobs", methods=["GET"])
def jobs():
    data = col.get(include=["metadatas"])
    return jsonify({"count": col.count(), "jobs": data["metadatas"]})


@app.route("/add", methods=["POST"])
def add():
    j = request.json or {}
    required = ["company", "title", "city", "type", "salary", "edu", "exp", "requirements"]
    if not all(k in j for k in required):
        return jsonify({"error": "缺少必填字段"}), 400
    jobs = json.load(open("jobs.json", encoding="utf-8"))
    new_id = max([x["id"] for x in jobs]) + 1
    j["id"] = new_id
    j.setdefault("tags", [])
    jobs.append(j)
    json.dump(jobs, open("jobs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    text = job_to_text(j)
    emb = embed_query(text)
    col.add(
        ids=[str(new_id)], documents=[text], embeddings=[emb],
        metadatas=[{
            "company": j["company"], "title": j["title"], "city": j["city"],
            "salary": j["salary"], "type": j["type"], "tags": ",".join(j.get("tags", [])),
        }],
    )
    return jsonify({"ok": True, "id": new_id, "count": col.count()})


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
