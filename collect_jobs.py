# -*- coding: utf-8 -*-
"""
岗位采集 / 扩充脚本（并发版）
====================
用法：
  python collect_jobs.py --target 2000
      基于「现有真实岗位 + 广度种子公司」，用硅基流动 DeepSeek 并发生成更多结构化岗位，
      写入 jobs.json。生成项 source 标为 "ai-augmented"，与人工核校的真实采集项区分。
  python collect_jobs.py --target 2000 --workers 8 --per-call 12
      提高并发 worker 数与每调用变体数，进一步提速（注意免费额度下的限流）。
  python collect_jobs.py --target 2000 --dry-run
      只统计将要生成多少条，不落盘。

设计说明（重要）：
  - 真实岗位（source 为 猎聘/实习僧/智联/BOSS/高校就业网 等）来自人工核校，是「主数据」。
  - ai 模式用于扩充样本量、丰富问法覆盖，属于「AI 辅助合成扩充」，并非已核实招聘信息，
    请勿直接据此投递；它们让 RAG 在更多问法下能命中、演示更稳。
  - 真实招聘站点（Boss/猎聘/智联）有反爬与登录限制，自动爬取需逐站适配，本脚本不含。
"""
import os
import sys
import json
import time
import queue
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import API_KEY, make_client, CITY_BASE

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_PATH = os.path.join(HERE, "jobs.json")

# 广度种子：现有真实岗位之外，补充真实公司的城市 × 方向覆盖
SEEDS_BREADTH = [
    ("阿里巴巴", "杭州", "算法工程师实习"),
    ("网易", "杭州", "数据分析实习"),
    ("蚂蚁集团", "杭州", "机器学习实习"),
    ("海康威视", "杭州", "计算机视觉实习"),
    ("大华股份", "杭州", "前端开发实习"),
    ("字节跳动", "杭州", "数据开发实习"),
    ("新华三", "杭州", "网络研发实习"),
    ("小米", "南京", "软件研发实习"),
    ("苏宁", "南京", "数据分析实习"),
    ("中电科二十八所", "南京", "软件研发实习"),
    ("华为", "南京", "算法工程师实习"),
    ("科大讯飞", "苏州", "自然语言处理实习"),
    ("思必驰", "苏州", "语音算法实习"),
    ("同程旅行", "苏州", "后端开发实习"),
    ("科沃斯", "苏州", "嵌入式开发实习"),
    ("药明康德", "苏州", "数据管理实习"),
    ("微软", "苏州", "软件工程师实习"),
    ("博世", "苏州", "数据科学实习"),
    ("美团", "上海", "算法工程师实习"),
    ("拼多多", "上海", "数据分析实习"),
    ("携程", "上海", "后端开发实习"),
    ("哔哩哔哩", "上海", "前端开发实习"),
    ("小红书", "上海", "机器学习实习"),
    ("蔚来", "上海", "数据开发实习"),
    ("上汽集团", "上海", "车联网算法实习"),
    ("商汤科技", "上海", "计算机视觉实习"),
    ("地平线", "上海", "自动驾驶算法实习"),
    ("依图科技", "上海", "大模型算法实习"),
    ("远景能源", "无锡", "数据开发实习"),
    ("阿斯利康", "无锡", "生物统计实习"),
    ("雪浪数制", "无锡", "工业大数据实习"),
    ("朗新科技", "无锡", "前端开发实习"),
    ("先导智能", "无锡", "算法工程师实习"),
    ("恒力集团", "苏州", "数据分析实习"),
    ("天合光能", "常州", "数据开发实习"),
    ("理想汽车", "常州", "自动驾驶实习"),
    ("比亚迪", "常州", "嵌入式开发实习"),
    ("星宇股份", "常州", "软件研发实习"),
    ("罗莱生活", "南通", "数据分析实习"),
    ("中天科技", "南通", "数据开发实习"),
    ("通富微电", "南通", "算法工程师实习"),
    ("携程", "南通", "后端开发实习"),
    ("吉利汽车", "宁波", "智能驾驶实习"),
    ("舜宇光学", "宁波", "计算机视觉实习"),
    ("公牛集团", "宁波", "数据分析实习"),
    ("雅戈尔", "宁波", "数据运营实习"),
    ("宁波银行", "宁波", "数据分析实习"),
]


def load_jobs():
    with open(JOBS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_jobs(jobs):
    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)


def build_seeds(existing):
    """从现有岗位抽取 (company, city, title) 作种子，再并入广度种子，去重。"""
    seeds = []
    seen = set()
    for j in existing:
        key = (j["company"], j["city"], j["title"])
        if key not in seen:
            seen.add(key)
            seeds.append(key)
    for s in SEEDS_BREADTH:
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


def parse_variants(text):
    """从模型输出稳健解析 JSON 数组。"""
    t = text.strip()
    if "```" in t:
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    a, b = t.find("["), t.rfind("]")
    if a == -1 or b == -1:
        return []
    try:
        arr = json.loads(t[a:b + 1])
    except Exception:
        return []
    return arr if isinstance(arr, list) else []


def call_llm(client, model, content, retries=3):
    """带退避重试的 LLM 调用，瞬时 429/超时自动重试。"""
    last = None
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.9,
            )
        except Exception as e:
            last = e
            print(f"    ⚠ 调用失败({i+1}/{retries})：{e}", flush=True)
            time.sleep(2 * (i + 1))
    raise last


def make_prompt(company, city, title, k):
    return (
        f"请基于以下公司/城市/岗位，生成 {k} 条「彼此不同」的实习或校招岗位变体"
        f"（同一公司同一城市，但职责方向、技能要求、薪资各异）。\n"
        f"只输出一个 JSON 数组，不要解释、不要 markdown 围栏。\n"
        f"每个元素字段：company(字符串), title(字符串), city(固定为\"{city}\"), "
        f"type(实习/校招/社招 之一), salary(如\"200-250元/天\"或\"15-20K/月\"), "
        f"edu(本科/硕士/大专/不限), exp(如\"3个月\"/\"0经验\"/\"1年\"), "
        f"requirements(中文一句，<=120字，描述技能与职责), tags(2-5个中文词组成的数组)。\n"
        f"参考基准：company={company}, city={city}, title={title}。"
    )


def worker(q, jobs_out, lock, stop, counter, need, per, seen_keys, next_id_box):
    """从种子队列取任务，生成变体并去重入队；达到 need 即触发停止。"""
    try:
        client = make_client()
    except Exception as e:
        print(f"  ✗ worker 初始化失败（无 key？）：{e}", flush=True)
        stop.set()
        return
    model = "deepseek-ai/DeepSeek-V3"
    while not stop.is_set():
        try:
            seed = q.get(timeout=2)
        except queue.Empty:
            break
        company, city, title = seed
        try:
            r = call_llm(client, model, make_prompt(company, city, title, per))
            variants = parse_variants(r.choices[0].message.content or "")
        except Exception as e:
            print(f"  ✗ [{company}/{city}] 生成失败：{e}", flush=True)
            q.task_done()
            continue

        local = []
        for v in variants:
            if stop.is_set():
                break
            try:
                company_v = str(v.get("company", company)).strip() or company
                title_v = str(v.get("title", title)).strip() or title
                city_v = str(v.get("city", city)).strip() or city
                if not any(city_v.startswith(c) for c in CITY_BASE):
                    city_v = city
                req_v = str(v.get("requirements", "")).strip()
                if not req_v:
                    continue
                key = (company_v, title_v, city_v, req_v[:30])
                tags = v.get("tags", [])
                if isinstance(tags, str):
                    tags = [t for t in tags.replace("，", ",").split(",") if t.strip()][:5]
                job = {
                    "id": 0,  # 占位，主线程统一分配
                    "company": company_v, "title": title_v, "city": city_v,
                    "type": str(v.get("type", "实习")).strip() or "实习",
                    "salary": str(v.get("salary", "面议")).strip(),
                    "edu": str(v.get("edu", "本科")).strip(),
                    "exp": str(v.get("exp", "不限")).strip(),
                    "requirements": req_v,
                    "tags": tags[:5],
                    "source": "ai-augmented",
                    "updated": "2026-08",
                }
                local.append((key, job))
            except Exception:
                continue

        with lock:
            for key, job in local:
                if stop.is_set():
                    break
                if key in seen_keys:
                    continue
                if counter["n"] >= need:
                    stop.set()
                    break
                seen_keys.add(key)
                counter["n"] += 1
                job["id"] = next_id_box["v"]
                next_id_box["v"] += 1
                jobs_out.append(job)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000, help="目标总条数（含现有）")
    ap.add_argument("--per-call", type=int, default=12, dest="per_call",
                    help="每调用生成变体数（默认 12，控成本/质量）")
    ap.add_argument("--per-seed", type=int, dest="per_call",
                    help="（兼容旧参数）同 --per-call")
    ap.add_argument("--workers", type=int, default=6, help="并发 worker 数（默认 6）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    args = ap.parse_args()

    if not API_KEY:
        print("⚠️ 未设置 SILICONFLOW_API_KEY，无法调用大模型生成。")
        print("   复制 .env.example 为 .env 并填入 key 后重试。")
        sys.exit(1)

    jobs = load_jobs()
    current = len(jobs)
    need = max(0, args.target - current)
    if need == 0:
        print(f"已有 {current} 条，已达目标 {args.target}，无需扩充。")
        return

    seeds = build_seeds(jobs)
    random.shuffle(seeds)

    if args.dry_run:
        print(f"[dry-run] 当前 {current} 条，目标 {args.target}，需新增 {need}；"
              f"可用种子 {len(seeds)} 个，并发 {args.workers} worker。")
        return

    q = queue.Queue()
    for s in seeds:
        q.put(s)
    seen_keys = set(
        (j["company"], j["title"], j["city"], j.get("requirements", "")[:30]) for j in jobs
    )
    jobs_out = []
    counter = {"n": 0}
    next_id_box = {"v": max((j["id"] for j in jobs), default=0) + 1}
    lock = threading.Lock()
    stop = threading.Event()

    print(f"当前 {current} 条，目标 {args.target}，需新增 {need}")
    print(f"并发 {args.workers} worker，每调用生成 {args.per_call} 变体，种子 {len(seeds)} 个\n")
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(worker, q, jobs_out, lock, stop, counter, need, args.per_call, seen_keys, next_id_box)
            for _ in range(args.workers)
        ]
        try:
            while not stop.is_set():
                time.sleep(5)
                with lock:
                    n = counter["n"]
                print(f"  进度 {n}/{need}（{int(time.time()-start)}s）", flush=True)
                if n >= need:
                    break
        except KeyboardInterrupt:
            stop.set()
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"  ⚠ worker 异常：{e}", flush=True)

    jobs.extend(jobs_out)
    save_jobs(jobs)
    print(f"\n完成：新增 {len(jobs_out)} 条，jobs.json 现共 {len(jobs)} 条（用时 {int(time.time()-start)}s）。")
    print("提示：数据已更新，请运行 `python build_index.py` 或界面点「重建索引」刷新向量库。")


if __name__ == "__main__":
    main()
