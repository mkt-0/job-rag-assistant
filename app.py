"""岗位 RAG 交互服务：Flask + Chroma(硅基流动 bge-m3) + 硅基流动 LLM。
路由：
  GET  /          交互界面
  POST /ask      自然语言问答 -> {answer, sources}
  GET  /jobs     当前库岗位列表与数量
  GET  /stats    按城市/类型/标签的聚合统计
  POST /add      自助补充一条岗位（粘 JD 入库）
  POST /rebuild  用 jobs.json 重建向量索引（需 API key）
"""
import os
import json
import chromadb
from flask import Flask, request, jsonify, send_from_directory

from rag_core import (
    API_KEY, EMBED_MODEL, LLM_MODEL, CHROMA_DIR, COLLECTION,
    make_client, embed_query, job_to_text, job_meta, load_jobs, build_collection,
)

app = Flask(__name__, static_folder=".", template_folder=".")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
col = chroma.get_or_create_collection(name=COLLECTION)


def ensure_index():
    """启动时若向量库为空则自动构建（jobs.json 有数据的情况下）。"""
    try:
        if col.count() == 0:
            jobs = load_jobs()
            if jobs:
                print(f"[启动] 向量库为空，自动从 jobs.json 构建 {len(jobs)} 条...")
                build_collection(chroma_dir=CHROMA_DIR, verbose=False)
                print(f"[启动] 构建完成，当前 {col.count()} 条。")
    except Exception as e:
        print(f"[启动] 自动构建跳过（可能缺少 API key）：{e}")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    q = (request.json or {}).get("question", "").strip()
    if not q:
        return jsonify({"error": "问题为空"}), 400
    try:
        client = make_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    q_emb = embed_query(client, q)
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


@app.route("/stats", methods=["GET"])
def stats():
    import collections
    data = col.get(include=["metadatas"])
    metas = data["metadatas"]
    by_city = collections.Counter(m["city"] for m in metas)
    by_type = collections.Counter(m["type"] for m in metas)
    by_tag = collections.Counter()
    for m in metas:
        for t in (m.get("tags") or "").split(","):
            t = t.strip()
            if t:
                by_tag[t] += 1
    return jsonify({
        "total": len(metas),
        "by_city": dict(by_city.most_common()),
        "by_type": dict(by_type),
        "top_tags": dict(by_tag.most_common(15)),
    })


@app.route("/add", methods=["POST"])
def add():
    j = request.json or {}
    required = ["company", "title", "city", "type", "salary", "edu", "exp", "requirements"]
    if not all(k in j for k in required):
        return jsonify({"error": "缺少必填字段"}), 400
    try:
        client = make_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    jobs = load_jobs()
    new_id = max([x["id"] for x in jobs]) + 1 if jobs else 1
    j["id"] = new_id
    j.setdefault("tags", [])
    j.setdefault("source", "手动补充")
    j.setdefault("updated", "2026-08")
    jobs.append(j)
    with open("jobs.json", "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)

    text = job_to_text(j)
    emb = embed_query(client, text)
    col.add(
        ids=[str(new_id)], documents=[text], embeddings=[emb],
        metadatas=[job_meta(j)],
    )
    return jsonify({"ok": True, "id": new_id, "count": col.count()})


@app.route("/rebuild", methods=["POST"])
def rebuild():
    try:
        build_collection(chroma_dir=CHROMA_DIR)
        return jsonify({"ok": True, "count": col.count()})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


if __name__ == "__main__":
    ensure_index()
    app.run(host="0.0.0.0", port=5000, debug=False)
