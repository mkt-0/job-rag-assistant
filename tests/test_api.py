# -*- coding: utf-8 -*-
"""API 冒烟测试：不依赖 LLM/网络，测接口结构与无 key 报错路径。
运行：pytest -q
（需要 flask / chromadb / openai 已安装；/stats、/jobs 使用本地空库即可。）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_core
import app as appmod

client = appmod.app.test_client()


def test_stats_structure():
    r = client.get("/stats")
    assert r.status_code == 200
    d = r.get_json()
    for k in ("total", "by_city", "by_type", "top_tags"):
        assert k in d
    assert isinstance(d["total"], int)


def test_jobs_structure():
    r = client.get("/jobs")
    assert r.status_code == 200
    d = r.get_json()
    assert "count" in d and "jobs" in d
    assert isinstance(d["jobs"], list)


def test_ask_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 400


def test_ask_stream_empty_question():
    r = client.post("/ask_stream", json={"question": "   "})
    assert r.status_code == 400


def test_ask_no_api_key_returns_error(monkeypatch):
    # 模拟未配置 key：make_client 应抛 RuntimeError -> 500 + error 字段
    monkeypatch.setattr(rag_core, "API_KEY", "")
    r = client.post("/ask", json={"question": "无锡 数据开发 实习"})
    assert r.status_code == 500
    assert "error" in r.get_json()
