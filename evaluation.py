# -*- coding: utf-8 -*-
"""RAG 检索质量评估：对比「混合检索(RRF)基线」与「+bge-reranker 重排」。

- 评测集：覆盖城市 × 方向的自然语言问题，expect 为应出现在 top5 资料中的关键词。
- 指标1 命中率 hit@5：top5 资料是否覆盖全部 expect 关键词（关键词出现即算命中）。
- 指标2 平均相关性分(reranker 裁判)：用 reranker 给返回的 top5 逐一打分取均值，跨问题平均，
       作为相对相关性的客观裁判（避免人肉判断）。
- 仅需嵌入 + reranker API；无 SILICONFLOW_API_KEY 时自动退出并提示。

用法：python evaluation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (
    API_KEY, CHROMA_DIR, COLLECTION, make_client, build_collection,
    retrieve, rerank_scores,
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
    {"q": "无锡本科能投的 Java 后端实习", "expect": ["Java", "无锡", "本科"]},
    {"q": "南京硕士要求的机器学习算法岗", "expect": ["机器学习", "南京", "硕士"]},
    {"q": "苏州工业互联网或数字孪生方向的实习", "expect": ["工业", "苏州"]},
    {"q": "上海的 NLP 或自然语言处理实习机会", "expect": ["NLP", "自然语言", "上海"]},
    {"q": "合肥或长三角的新能源数据岗", "expect": ["新能源", "数据"]},
    {"q": "社招里三年经验的资深前端工程师薪资", "expect": ["前端", "社招"]},
    {"q": "医药/生物统计方向的实习在无锡", "expect": ["生物", "无锡"]},
    {"q": "计算机视觉实习需要 PyTorch 吗", "expect": ["计算机视觉", "PyTorch"]},
]


def mean_relevance(q, docs):
    """用 reranker 作为裁判，给一组文档打平均相关性分（0~1）。"""
    if not docs:
        return 0.0
    sc = rerank_scores(q, docs)
    if not sc:
        return 0.0
    return sum(sc.get(i, 0.0) for i in range(len(docs))) / len(docs)


def main():
    if not API_KEY:
        print("⚠️ 未设置 SILICONFLOW_API_KEY，跳过评估（嵌入 + reranker 需要 key）。")
        print("   复制 .env.example 为 .env 并填入 key 后重试。")
        return

    client = make_client()
    cdb = chromadb.PersistentClient(path=CHROMA_DIR)
    col = cdb.get_or_create_collection(name=COLLECTION)
    if col.count() == 0:
        print("向量库为空，先用 jobs.json 构建索引…")
        col = build_collection(client=client, verbose=False)

    print(f"库内岗位：{col.count()} 条，评测问题：{len(EVAL)} 个\n")
    hit_base = hit_re = 0
    rel_base_sum = rel_re_sum = 0.0
    for i, item in enumerate(EVAL, 1):
        docs_base, _ = retrieve(client, col, item["q"], n=5, rerank=False)
        docs_re, _ = retrieve(client, col, item["q"], n=5, rerank=True)
        blob_base = " ".join(docs_base)
        blob_re = " ".join(docs_re)
        miss_base = [k for k in item["expect"] if k not in blob_base]
        miss_re = [k for k in item["expect"] if k not in blob_re]
        ok_base = len(miss_base) == 0
        ok_re = len(miss_re) == 0
        hit_base += ok_base
        hit_re += ok_re
        rel_base = mean_relevance(item["q"], docs_base)
        rel_re = mean_relevance(item["q"], docs_re)
        rel_base_sum += rel_base
        rel_re_sum += rel_re
        tag = "✅" if (ok_base or ok_re) else "❌"
        print(f"{tag} [{i}] {item['q']}")
        if not ok_base:
            print(f"     基线未命中关键词：{miss_base}")
        if not ok_re:
            print(f"     重排未命中关键词：{miss_re}")
        print(f"     相关性分  基线={rel_base:.3f}   重排={rel_re:.3f}")

    n = len(EVAL)
    print(f"\n========== 评估结果（{n} 题）==========")
    print(f"命中率 hit@5        ：基线 {hit_base}/{n} ({hit_base/n*100:.1f}%)  →  重排 {hit_re}/{n} ({hit_re/n*100:.1f}%)")
    print(f"平均相关性分(裁判)  ：基线 {rel_base_sum/n:.3f}  →  重排 {rel_re_sum/n:.3f}")
    print(f"相关性分提升        ：{rel_re_sum/n - rel_base_sum/n:+.3f}")


if __name__ == "__main__":
    main()
