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
