# -*- coding: utf-8 -*-
"""RAG 检索质量评估：用一组标注问题测「召回@5」。

- 每条评测项含一个自然语言问题 q 与若干期望命中关键词 expect（出现在 top5 检索文本中即算命中）。
- 不依赖 LLM 生成，只测「检索」这一步的命中率，可离线（仅需嵌入 API）。
- 用法：python evaluation.py
- 无 SILICONFLOW_API_KEY 时自动退出并提示。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (
    API_KEY, CHROMA_DIR, COLLECTION, make_client, build_collection,
    load_jobs, embed_query, job_to_text,
)
import chromadb


# 标注评测集：覆盖城市 × 方向，expect 为应出现在 top5 资料中的关键词
EVAL = [
    {"q": "无锡有哪些数据开发相关的实习？", "expect": ["数据开发", "无锡"]},
    {"q": "苏州的测试开发实习要求会什么？", "expect": ["测试开发", "苏州"]},
    {"q": "南京前端开发实习用什么框架？", "expect": ["前端", "南京", "Vue"]},
    {"q": "上海大模型算法实习做什么？", "expect": ["大模型", "上海"]},
    {"q": "杭州数据分析实习需要 SQL 吗？", "expect": ["数据分析", "杭州", "SQL"]},
    {"q": "宁波有没有大模型或智能体方向的算法实习？", "expect": ["大模型", "宁波"]},
    {"q": "常州的全栈或前端实习机会", "expect": ["前端", "全栈", "常州"]},
    {"q": "南通的数据分析实习", "expect": ["数据分析", "南通"]},
    {"q": "嵌入式软件实习在无锡好找吗？", "expect": ["嵌入式", "无锡"]},
    {"q": "面向本科生的人工智能产品实习在上海", "expect": ["产品", "上海"]},
    {"q": "校招里算法工程师有哪些公司", "expect": ["算法", "校招"]},
    {"q": "用 Python 做数据分析的实习岗位", "expect": ["Python", "数据分析"]},
]


def main():
    if not API_KEY:
        print("⚠️ 未设置 SILICONFLOW_API_KEY，跳过评估（嵌入需要 key）。")
        print("   复制 .env.example 为 .env 并填入 key 后重试。")
        return

    client = make_client()
    cdb = chromadb.PersistentClient(path=CHROMA_DIR)
    col = cdb.get_or_create_collection(name=COLLECTION)
    if col.count() == 0:
        print("向量库为空，先用 jobs.json 构建索引…")
        col = build_collection(client=client, verbose=False)

    print(f"库内岗位：{col.count()} 条，评测问题：{len(EVAL)} 个\n")
    hit = 0
    for i, item in enumerate(EVAL, 1):
        q_emb = embed_query(client, item["q"])
        res = col.query(query_embeddings=[q_emb], n_results=5)
        docs = res["documents"][0]
        blob = " ".join(docs)
        missing = [k for k in item["expect"] if k not in blob]
        ok = len(missing) == 0
        hit += 1 if ok else 0
        status = "✅" if ok else "❌"
        print(f"{status} [{i}] {item['q']}")
        if not ok:
            print(f"     未命中关键词：{missing}")
        # 展示 top1 来源
        top = docs[0].split(" | ")[0] if docs else "（无）"
        print(f"     top1: {top}")

    print(f"\n召回@5：{hit}/{len(EVAL)} = {hit/len(EVAL)*100:.1f}%")


if __name__ == "__main__":
    main()
