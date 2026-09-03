"""岗位 RAG 交互服务：Flask + Chroma(硅基流动 bge-m3) + 硅基流动 LLM。
路由：
  GET  /           交互界面
  GET  /health     健康检查(无需 key)
  POST /ask        单轮问答(非流式) -> {answer, sources}
  POST /ask_stream 单轮问答(SSE 流式) -> text/event-stream
  POST /chat       多轮对话(非流式) -> {session_id, answer, sources, history}
  POST /chat_stream 多轮对话(SSE 流式) -> text/event-stream
  GET  /jobs       当前库岗位列表与数量
  GET  /stats      按城市/类型/标签的聚合统计
  POST /add        自助补充一条岗位(粘 JD 入库)
  POST /rebuild    用 jobs.json 重建向量索引(需 API key)
稳定性：所有业务路由均包 try/except 返回结构错误；生成失败走降级(返回检索岗位)。
"""
import os
import json
import collections
import secrets
import threading
import chromadb
from flask import Flask, request, jsonify, send_from_directory, Response

from rag_core import (
    API_KEY, CHROMA_DIR, COLLECTION, CITY_BASE, normalize_city, normalize_edu,
    make_client, embed_query, job_to_text, job_meta, load_jobs, build_collection,
    retrieve, rewrite_query, build_messages, stream_answer, fallback_answer, resolve_model,
)

app = Flask(__name__, static_folder=".", template_folder=".")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
col = chroma.get_or_create_collection(name=COLLECTION)

# 多轮会话内存(进程内，重启即清空；演示足够)
sessions = {}

# 重建索引时加锁，避免并发读写冲突
_rebuild_lock = threading.Lock()


def _safe_col():
    """取出 collection；若句柄失效(如被外部删除)则自动重新获取。"""
    global col
    try:
        col.count()
    except Exception:
        col = chroma.get_or_create_collection(name=COLLECTION)
    return col


def ensure_index():
    """启动时若向量库为空则自动构建(jobs.json 有数据的情况下)。"""
    try:
        if col.count() == 0:
            jobs = load_jobs()
            if jobs:
                print(f"[启动] 向量库为空，自动从 jobs.json 构建 {len(jobs)} 条...")
                build_collection(chroma_dir=CHROMA_DIR, verbose=False)
                print(f"[启动] 构建完成，当前 {col.count()} 条。")
    except Exception as e:
        print(f"[启动] 自动构建跳过(可能缺少 API key 或网络)：{e}")


def _sources_from(metas):
    return [
        {
            "company": m["company"], "title": m["title"], "city": m["city"],
            "salary": m["salary"], "type": m["type"], "tags": m.get("tags", ""),
            "edu": m.get("edu", ""), "exp": m.get("exp", ""),
            "requirements": m.get("requirements", ""), "source": m.get("source", ""),
        }
        for m in metas
    ]


@app.route("/health")
def health():
    try:
        cnt = col.count()
    except Exception:
        cnt = 0
    return jsonify({
        "status": "ok",
        "has_key": bool(API_KEY),
        "index_count": cnt,
    })


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "问题为空"}), 400
    try:
        client = make_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        model = resolve_model(data.get("model"))
        smart = bool(data.get("smart"))
        filters = data.get("filters") or {}
        q_search = rewrite_query(client, q) if smart else q

        metas = []
        docs, metas = retrieve(client, col, q_search, n=5, filters=filters)
        if not docs:
            return jsonify({"answer": "资料库中暂无相关岗位，换个关键词试试？", "sources": []})
        context = "\n\n".join(docs)
        messages = build_messages([], context, q)
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
        answer = resp.choices[0].message.content
        return jsonify({"answer": answer, "sources": _sources_from(metas)})
    except Exception as e:
        # 降级：生成失败但检索到了资料（复用已检索 metas，避免重复嵌入）
        if metas:
            return jsonify({"answer": fallback_answer(metas),
                            "sources": _sources_from(metas), "degraded": True})
        return jsonify({"error": f"处理失败：{e}"}), 500


@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    data = request.json or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "问题为空"}), 400
    try:
        client = make_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    model = resolve_model(data.get("model"))
    smart = bool(data.get("smart"))
    filters = data.get("filters") or {}

    def gen():
        try:
            q_search = rewrite_query(client, q) if smart else q
            docs, metas = retrieve(client, col, q_search, n=5, filters=filters)
            sources = _sources_from(metas)
            yield "data: " + json.dumps({"type": "sources", "sources": sources},
                                        ensure_ascii=False) + "\n\n"
            if not docs:
                yield "data: " + json.dumps({"type": "token",
                                             "token": "资料库中暂无相关岗位，换个关键词试试？"},
                                            ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
                return
            context = "\n\n".join(docs)
            messages = build_messages([], context, q)
            try:
                for tok in stream_answer(client, model, messages):
                    yield "data: " + json.dumps({"type": "token", "token": tok},
                                                ensure_ascii=False) + "\n\n"
            except Exception as e:
                # 生成中途失败 -> 降级为列出资料
                yield "data: " + json.dumps({"type": "token",
                                             "token": "\n\n" + fallback_answer(metas)},
                                            ensure_ascii=False) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "error": str(e)},
                                        ensure_ascii=False) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"

    return Response(gen(), mimetype="text/event-stream")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "问题为空"}), 400
    try:
        client = make_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        sid = data.get("session_id") or secrets.token_hex(8)
        model = resolve_model(data.get("model"))
        smart = bool(data.get("smart"))
        filters = data.get("filters") or {}
        hist = sessions.get(sid, [])

        q_search = rewrite_query(client, q) if smart else q
        metas = []
        docs, metas = retrieve(client, col, q_search, n=5, filters=filters)
        if not docs:
            return jsonify({"session_id": sid, "answer": "资料库中暂无相关岗位，换个关键词试试？",
                            "sources": [], "history": hist})
        context = "\n\n".join(docs)
        messages = build_messages(hist, context, q)
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
        answer = resp.choices[0].message.content

        hist.append({"role": "user", "content": q})
        hist.append({"role": "assistant", "content": answer})
        sessions[sid] = hist[-12:]

        return jsonify({
            "session_id": sid, "answer": answer,
            "sources": _sources_from(metas), "history": sessions[sid],
        })
    except Exception as e:
        if metas:
            return jsonify({"session_id": sid,
                            "answer": fallback_answer(metas),
                            "sources": _sources_from(metas), "degraded": True})
        return jsonify({"error": f"处理失败：{e}"}), 500


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.json or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "问题为空"}), 400
    try:
        client = make_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    sid = data.get("session_id") or secrets.token_hex(8)
    model = resolve_model(data.get("model"))
    smart = bool(data.get("smart"))
    filters = data.get("filters") or {}

    def gen():
        try:
            q_search = rewrite_query(client, q) if smart else q
            docs, metas = retrieve(client, col, q_search, n=5, filters=filters)
            sources = _sources_from(metas)
            yield "data: " + json.dumps({"type": "sources", "sources": sources,
                                         "session_id": sid}, ensure_ascii=False) + "\n\n"
            if not docs:
                yield "data: " + json.dumps({"type": "token",
                                             "token": "资料库中暂无相关岗位，换个关键词试试？"},
                                            ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps({"type": "done", "session_id": sid},
                                            ensure_ascii=False) + "\n\n"
                return
            context = "\n\n".join(docs)
            messages = build_messages(sessions.get(sid, []), context, q)
            answer_buf = []
            try:
                for tok in stream_answer(client, model, messages):
                    answer_buf.append(tok)
                    yield "data: " + json.dumps({"type": "token", "token": tok},
                                                ensure_ascii=False) + "\n\n"
            except Exception:
                yield "data: " + json.dumps({"type": "token",
                                             "token": "\n\n" + fallback_answer(metas)},
                                            ensure_ascii=False) + "\n\n"
            a = "".join(answer_buf)
            hh = sessions.get(sid, [])
            hh.append({"role": "user", "content": q})
            hh.append({"role": "assistant", "content": a})
            sessions[sid] = hh[-12:]
            yield "data: " + json.dumps({"type": "done", "session_id": sid},
                                        ensure_ascii=False) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "error": str(e)},
                                        ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "session_id": sid},
                                        ensure_ascii=False) + "\n\n"

    return Response(gen(), mimetype="text/event-stream")


@app.route("/jobs", methods=["GET"])
def jobs():
    try:
        c = _safe_col()
        data = c.get(include=["metadatas"])
        return jsonify({"count": c.count(), "jobs": data["metadatas"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        c = _safe_col()
        data = c.get(include=["metadatas"])
        metas = data["metadatas"]
        # 看板聚焦核心城市：非 CITY_BASE 的城市统一并入「其他」
        by_city = collections.Counter()
        for m in metas:
            cy = m["city"]
            if cy not in CITY_BASE:
                cy = "其他"
            by_city[cy] += 1
        by_type = collections.Counter(m["type"] for m in metas)
        by_edu = collections.Counter(normalize_edu(m.get("edu", "")) for m in metas)
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
            "by_edu": dict(by_edu.most_common()),
            "top_tags": dict(by_tag.most_common(15)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

    try:
        jobs = load_jobs()
        # 城市归一化：存入 jobs.json 与向量库前先清洗（如 "无锡新吴" -> "无锡"）
        j["city"] = normalize_city(j["city"])
        # 去重：公司+岗位+城市 完全相同则视为重复，直接返回已有记录
        key = (j["company"].strip(), j["title"].strip(), j["city"])
        for x in jobs:
            if (x["company"].strip(), x["title"].strip(), normalize_city(x["city"])) == key:
                return jsonify({"ok": True, "id": x["id"], "count": col.count(), "dup": True})
        new_id = max([x["id"] for x in jobs]) + 1 if jobs else 1
        j["id"] = new_id
        j.setdefault("tags", [])
        j.setdefault("source", "手动补充")
        j.setdefault("updated", "2026-09")
        jobs.append(j)
        with open("jobs.json", "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)

        text = job_to_text(j)
        emb = embed_query(client, text)
        col.add(ids=[str(new_id)], documents=[text], embeddings=[emb],
                metadatas=[job_meta(j)])
        return jsonify({"ok": True, "id": new_id, "count": col.count()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rebuild", methods=["POST"])
def rebuild():
    global col
    try:
        with _rebuild_lock:
            build_collection(chroma_dir=CHROMA_DIR)
            col = chroma.get_or_create_collection(name=COLLECTION)
        return jsonify({"ok": True, "count": col.count()})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


if __name__ == "__main__":
    ensure_index()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
