# 岗位 RAG 智能助手（job-rag-assistant）

面向大学生的**实习 / 校招 / 社招**岗位检索增强（RAG）问答系统。基于真实公开招聘数据，用向量检索 + 大语言模型，回答「某城市某方向的岗位要求 / 匹配度 / 投递建议」等问题。

> 技术栈：Flask + Chroma 向量库（硅基流动 `BAAI/bge-m3` 嵌入）+ 硅基流动 `DeepSeek-V3 / R1` 生成。
> 数据：长三角（无锡 / 苏州 / 南京 / 上海 / 常州 / 南通 / 杭州 / 宁波）岗位。当前 **1000+ 条**，其中真实核校样本（来源：猎聘 / 实习僧 / 智联 / 应届生求职网 / 高校就业网等）约 222 条，AI 辅助扩充约 800+ 条，所有条目均带 `source` 标注以便区分。

> **最近优化（v2）**：一键启动脚本 `run.bat` / `run.sh`；查询嵌入缓存（相似问题秒回）；混合检索（向量 + 中文 bigram 关键词重排）提升召回；客户端超时 + 自动重试抗限流；生成失败自动降级为「列出检索岗位」；新增 `/health` 健康检查；界面重做（流式打字、来源卡片、数据看板）。

---

## 功能特性

- **真·接入大模型**：检索到的岗位资料作为上下文喂给 LLM，生成结构化、可行动的答案（不是简单关键词匹配）。
- **一键启动**：`run.bat`（Windows 双击）/ `run.sh`（mac/Linux）自动建环境、装依赖、起服务、开浏览器，零命令。
- **更聪明的检索**：向量相似 + 中文 bigram 关键词信号混合重排（RRF 思路），top-k 召回更准；可叠加「LLM 改写问题」智能检索。
- **更快**：查询嵌入结果进程内缓存，相似问题直接命中，省去重复 API 调用。
- **更稳**：客户端超时 + 自动重试抗限流；生成失败自动降级为「列出检索到的岗位」，保证永远有结果；新增 `/health` 健康检查。
- **多轮对话**：`/chat` 与 `/chat_stream` 维护会话上下文，可追问、对比、深挖。
- **流式输出**：SSE 流式打字效果，回答逐字出现。
- **模型可切换**：标准 `DeepSeek-V3`（快） / 深度思考 `DeepSeek-R1`（慢但推理更强）。
- **数据看板**：页面顶部实时展示岗位总数、城市 / 类型 / 方向分布。
- **自助扩充**：`/add` 接口粘贴 JD 即可入库；`/rebuild` 一键重建索引；`collect_jobs.py` 批量扩充。
- **质量保障**：`evaluation.py` 测检索召回，`tests/` 下 pytest 冒烟测试，CI 自动回归。

---

## 目录结构

```
.
├── run.bat           # Windows 一键启动（建环境/装依赖/起服务/开浏览器）
├── run.sh            # macOS / Linux 一键启动
├── app.py            # Flask 服务：/ /health /ask /ask_stream /chat /chat_stream /stats /jobs /add /rebuild
├── rag_core.py       # 共享核心：客户端(超时重试)/嵌入缓存/混合重排检索/LLM 对话/降级
├── build_index.py    # 用 jobs.json 重建 Chroma 向量索引
├── collect_jobs.py   # 用大模型批量扩充岗位数据（写入 jobs.json，source=ai-augmented）
├── index.html        # 单页交互界面（流式打字 + 多轮 + 模型切换 + 看板 + 来源卡片）
├── jobs.json         # 岗位数据（1000+ 条，字段规范，含 source 标注）
├── evaluation.py     # 检索召回@5 评估（标注问题集）
├── tests/test_api.py # pytest 冒烟测试（接口结构 + 无 key 报错路径）
├── .github/workflows/eval.yml  # CI：push/PR 跑 pytest + 可选召回评估
├── requirements.txt
├── .env.example      # 环境变量模板
└── README.md
```

---

## 快速开始

### 方式一：一键启动（推荐，零命令）

- **Windows**：双击 `run.bat`。它会自动建虚拟环境、装依赖、启动服务，等索引就绪后自动打开浏览器 `http://localhost:5000`。
- **macOS / Linux**：`bash run.sh`（或 `chmod +x run.sh && ./run.sh`）。

### 方式二：手动

```bash
# 1. 准备环境（Python 3.10+）
pip install -r requirements.txt

# 2. 配置密钥：复制模板并填入你的硅基流动 key
cp .env.example .env
#   编辑 .env：SILICONFLOW_API_KEY=sk-xxxx

# 3. 启动（首次会自动用 jobs.json 构建向量索引，需联网调用嵌入 API）
python app.py

# 4. 浏览器打开
http://localhost:5000
```

> 不要用 `file://` 双击打开 `index.html`——浏览器会拦截 `fetch` 请求，必须通过 `http://localhost:5000` 访问。

### 健康检查

```bash
curl http://localhost:5000/health
# -> {"status":"ok","has_key":true,"index_count":1000+}
```

---

## 数据从哪来

- 来源：猎聘、实习僧、智联校园、应届生求职网、BOSS 直聘聚合、全职招聘网、得早学就创、牛客、高校就业服务平台、企业官网等**公开渠道**检索整理。
- 诚实标注：每条岗位 `source` 字段记录检索来源；`updated` 记录整理月份。
- 规模与边界：真实核校样本约 222 条，覆盖长三角主要城市与主流技术方向（大数据 / 算法 / AI / 前端 / 测试 / 嵌入式 / 数据分析 / 产品 等）。其余为 **AI 辅助扩充**（`source="ai-augmented"`）——基于真实公司/岗位分布由大模型生成，用于提升样本量与问法覆盖，**非已核实招聘信息，请勿据此直接投递**。

### 用脚本批量扩充（`collect_jobs.py`）

```bash
# 基于现有真实岗位 + 广度种子公司，用 DeepSeek 生成更多结构化岗位，写入 jobs.json
python collect_jobs.py --target 1000

# 只预览将生成多少条，不落盘
python collect_jobs.py --target 1000 --dry-run

# 控制每种子变体数（默认按目标自动算，上限 6）
python collect_jobs.py --target 800 --per-seed 3
```

脚本会去重（公司/岗位/城市/要求相近则跳过）、校验字段、自动分配 `id`，并标注 `source="ai-augmented"`。运行需 `SILICONFLOW_API_KEY`。真实招聘站点（Boss/猎聘/智联）有反爬与登录限制，自动爬取需逐站适配，本脚本不含。

---

## 接口速览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 交互界面 |
| POST | `/ask` | 单轮问答，返回 `{answer, sources}` |
| POST | `/ask_stream` | 单轮流式（SSE，事件：`sources`/`token`/`done`/`error`） |
| POST | `/chat` | 多轮对话，返回 `{session_id, answer, sources, history}` |
| POST | `/chat_stream` | 多轮流式（SSE） |
| GET | `/stats` | 城市 / 类型 / 标签聚合统计 |
| GET | `/jobs` | 岗位列表与数量 |
| POST | `/add` | 自助补充一条岗位（粘 JD 入库） |
| POST | `/rebuild` | 用 `jobs.json` 重建向量索引 |

`/ask`、`/chat` 请求体示例：
```json
{ "question": "无锡有哪些适合本科的数据开发实习？",
  "model": "v3",          // v3=标准, r1=深度思考
  "smart": false,         // true=先 LLM 改写问题再检索
  "session_id": "" }      // 多轮时传上次返回的 session_id
```

---

## 质量评估与测试

```bash
# 检索召回评估（需 API key，离线嵌入；无 key 时自动跳过）
python evaluation.py

# 接口冒烟测试（无需 key，测结构 + 无 key 报错路径）
pytest -q
```

### CI（GitHub Actions）

仓库内置 `.github/workflows/eval.yml`：每次 push / PR 到 `main` 自动：
1. 安装依赖并跑 `pytest -q`（无需密钥）；
2. 若仓库设置了 **secret `SILICONFLOW_API_KEY`**，额外跑 `evaluation.py` 测检索召回@5（未设置则自动跳过，不阻断 CI）。

建议把你的硅基流动 key 配置为仓库 Secret，让每次提交都自动回归召回率。

---

## 安全

- 密钥只放在本地 `.env`，**不要提交**（已在 `.gitignore` 排除 `.env` 与 `chroma_db/`）。
- 多轮会话历史保存在进程内存，重启即清空，仅用于演示。
- 生成答案严格基于检索资料，缺失时如实说明「资料中未收录」，避免编造公司与薪资。

---

## 后续可扩展

- 接实时招聘 API（猎聘 / BOSS 开放能力或第三方）把真实样本扩到更大规模。
- 给 `collect_jobs.py` 增加真实站点适配器（需处理反爬 / 登录态）。
- 加用户反馈标注，用评估集持续监控召回与答案质量。
