# -*- coding: utf-8 -*-
"""核心归一化函数的单元测试（无需网络 / API key）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_core
import app as appmod


def test_normalize_city_compound():
    # 复合 / 噪声写法应归并到核心城市
    assert rag_core.normalize_city("无锡新吴") == "无锡"
    assert rag_core.normalize_city("全国(含上海)") == "上海"
    assert rag_core.normalize_city("北京/上海") == "上海"
    assert rag_core.normalize_city("南京/无锡/成都") == "无锡"  # 取 CITY_BASE 首个命中
    # 不在 CITY_BASE 的城市原样返回（统计时再归并「其他」）
    assert rag_core.normalize_city("深圳") == "深圳"


def test_normalize_edu_buckets():
    assert appmod.normalize_edu("本科") == "本科"
    assert appmod.normalize_edu("硕士(优秀本科)") == "硕士"
    assert appmod.normalize_edu("大专及以上") == "大专/中专"
    assert appmod.normalize_edu("不限") == "不限"
    assert appmod.normalize_edu("本科/硕士/博士") == "博士"  # 范围取最高


def test_build_where_none():
    assert rag_core.build_where(None) is None
    assert rag_core.build_where({}) is None
    assert rag_core.build_where({"city": "", "type": None}) is None


def test_build_where_single():
    assert rag_core.build_where({"city": "无锡"}) == {"city": "无锡"}
    assert rag_core.build_where({"type": "算法"}) == {"type": "算法"}


def test_build_where_combined():
    out = rag_core.build_where({"city": "无锡", "type": "算法", "edu": "本科"})
    assert out == {"$and": [
        {"city": "无锡"}, {"type": "算法"}, {"edu_norm": "本科"}
    ]}
    # 仅两项时仍是 $and
    out2 = rag_core.build_where({"city": "上海", "edu": "硕士"})
    assert out2 == {"$and": [{"city": "上海"}, {"edu_norm": "硕士"}]}


def test_rerank_docs_no_key_returns_none(monkeypatch):
    # 无 key 时 reranker 应安全降级为 None（由 retrieve 回退混合重排），不抛异常
    monkeypatch.setattr(rag_core, "API_KEY", "")
    assert rag_core.rerank_docs("无锡 数据开发", ["岗位A", "岗位B"]) is None
    # 空候选也应返回 None
    assert rag_core.rerank_docs("q", []) is None


def test_hybrid_rerank_orders_by_score():
    # 向量相似 + 关键词重叠：与 query 更相关的 "high" 应排在 "low" 之前
    docs = ["low", "high"]
    metas = [{"city": "x"}, {"city": "y"}]
    dists = [0.9, 0.2]   # 距离越大越不相似
    out_docs, out_metas = rag_core._hybrid_rerank("high", docs, metas, dists, 2)
    assert out_docs[0] == "high"
    assert out_metas[0] == {"city": "y"}
