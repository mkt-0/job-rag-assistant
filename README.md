# 岗位 RAG 智能助手（job-rag-assistant）

面向大学生的**实习 / 校招 / 社招**岗位检索增强（RAG）问答系统。基于真实公开招聘数据，用向量检索 + 大语言模型，回答「某城市某方向的岗位要求 / 匹配度 / 投递建议」等问题。

> 技术栈：Flask + Chroma 向量库（硅基流动 `BAAI/bge-m3` 嵌入）+ 硅基流动 `DeepSeek-V3 / R1` 生成。
> 数据：长三角（上海 / 南京 / 苏州 / 无锡 / 杭州 / 常州 / 南通 / 宁波 / 合肥 / 嘉兴 / 绍兴 / 扬州 / 镇江 / 泰州 / 盐城 / 湖州 / 金华 / 台州 / 温州）岗位。当前 **4200 条**，覆盖**全部职能类别**——技术研发、产品、设计、运营、市场/营销、销售、客服、人力资源、行政、财务/会计、法务/合规、供应链/物流、生产/制造、医疗/健康、教育培训、金融/银行、房地产/建筑/工程、媒体/内容、翻译/语言、餐饮/酒店/零售、综合/管培，共 21 类。其中技术类约 2000 条，其余 20 类各约 110 条；均带 `source` 与 `category` 标注。

> **最近优化（v7）**：**打破「只用计算机岗」的数据局限**——原数据采集种子 100% 是算法/前后端/ML 等计算机岗位，导致检索严重偏向技术类。新增 `expand_jobs.py` 离线生成器，覆盖**求职软件上能出现的全部职能类别**，并给每条岗位补齐 `category`（职能类别）字段；检索文本、元数据、筛选、看板统计、前端筛选条均贯通 `category`，现在可按「职能类别」硬筛选（如只看财务/医疗/销售岗）。生成确定性、可复现、不依赖 API。

> **最近优化（v6）**：项目体检与加固——**修复一处密钥泄露**：教学文件 `rag_minimal.py` 曾把真实 API key 明文硬编码且已推到公开仓库，已改为从 `.env` 读取（并建议轮换 key）；`/health` 改用实时密钥状态；`/add` 改用文档专用嵌入接口、不再污染查询缓存；`rewrite_query` 增加结果缓存（相同问题免再调一次 LLM）；新增 `rerank_docs` 降级与混合重排单测，pytest 增至 12 项全过。
>
> **最近优化（v4）**：数据看板**可交互筛选**——点击城市 / 用工类型 / 学历标签即按该条件硬筛选检索并自动提问，顶部出现可清除的筛选条（支持多条件 AND）；`/add` 新增去重与城市归一化；`build_where` 纯函数化便于单测；单测增至 10 项全过；`normalize_edu` 统一到核心模块。
>
> **最近优化（v5）**：新增 **bge-reranker 交叉编码器重排**——向量召回 top-20 候选经硅基流动 `BAAI/bge-reranker-v2-m3` 精排 top-5，top5 平均相关性分由 0.838 提升到 0.885（**+5.6%**）；重排失败自动降级为混合重排；评估脚本 `evaluation.py` 现已对比「基线 / 重排」两套方案并输出真实数字。
>
> **最近优化（v3）**：数据扩至 2000+ 条；参考岗位新增「真实核校 / AI扩充」**数据可信度标识**（诚实区分人工核校与模型合成）；来源卡片补全**经验要求**与**岗位要求摘要**；数据看板新增**学历分布**；后端移除冗余重排、合并统计聚合；补充核心归一化函数单元测试。

> **最近优化（v2）**：一键启动脚本 `run.bat` / `run.sh`；查询嵌入缓存（相似问题秒回）；混合检索（向量 + 中文 bigram 关键词重排）提升召回；客户端超时 + 自动重试抗限流；生成失败自动降级为「列出检索岗位」；新增 `/health` 健康检查；界面重做（流式打字、来源卡片、数据看板）。

---

## 功能特性

- **真·接入大模型**：检索到的岗位资料作为上下文喂给 LLM，生成结构化、可行动的答案（不是简单关键词匹配）。
- **一键启动**：`run.bat`（Windows 双击）/ `run.sh`（mac/Linux）自动建环境、装依赖、起服务、开浏览器，零命令。
- **冷启动零配置**：clone 后脚本自动生成 `.env`（从 `.env.example`）、首次启动自动用 `jobs.json` 构建向量索引；**未填密钥时给出明确指引而非跑出空库**。
- **更聪明的检索**：向量相似 + 中文 bigram 关键词信号混合重排（RRF 思路），top-k 召回更准；可叠加「LLM 改写问题」智能检索。
- **交叉编码器重排**：向量召回 top-20 候选经硅基流动 `BAAI/bge-reranker-v2-m3` 精排 top-5，检索精度更高；重排失败自动降级为混合重排，保证有结果（v5）。
- **更快**：查询嵌入与 LLM 改写结果均进程内缓存，相似问题直接命中，省去重复 API 调用。
- **更稳**：客户端超时 + 自动重试抗限流；生成失败自动降级为「列出检索到的岗位」，保证永远有结果；新增 `/health` 健康检查。
- **多轮对话**：`/chat` 与 `/chat_stream` 维护会话上下文，可追问、对比、深挖。
- **流式输出**：SSE 流式打字效果，回答逐字出现。
- **模型可切换**：标准 `DeepSeek-V3`（快） / 深度思考 `DeepSeek-R1`（慢但推理更强）。
- **数据看板**：页面顶部实时展示岗位总数、城市 / 类型 / **学历要求** / **职能类别** / 方向分布。
- **看板可交互筛选**：点击看板里的城市 / 用工类型 / 学历 / **职能类别**标签，即按该条件**硬筛选**检索并自动提问；顶部出现可清除的筛选条（支持多条件 AND 组合）。
- **数据可信度标识**：每个参考岗位标注「真实核校 / AI扩充」来源徽章，诚实区分人工核校公开渠道与模型合成样例，避免误导。
- **来源卡片增强**：展示公司、岗位、城市·薪资·**经验**、**要求摘要**、学历与方向标签，信息更完整。
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
├── jobs.json         # 岗位数据（2000 条，字段规范，含 source 标注）
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

- **Windows**：双击 `run.bat`。它会自动建虚拟环境、装依赖、**从 `.env.example` 生成 `.env`**、启动服务，等索引就绪后自动打开浏览器 `http://localhost:5000`。
- **macOS / Linux**：`bash run.sh`（或 `chmod +x run.sh && ./run.sh`）。
- ⚠️ 首次运行若看到「尚未配置有效的 SILICONFLOW_API_KEY」提示：打开 `.env` 把 `SILICONFLOW_API_KEY=在此填写你的key` 改成你的真实 key（https://cloud.siliconflow.cn 免费获取），保存后重新运行即可。

### 方式二：手动

```bash
# 1. 准备环境（Python 3.10+）
pip install -r requirements.txt

# 2. 配置密钥：首次运行会自动从 .env.example 生成 .env，你只需编辑填入真实 key
cp .env.example .env   # 若 .env 已存在可跳过；也可直接编辑已有 .env
#   把 SILICONFLOW_API_KEY=在此填写你的key 改成 SILICONFLOW_API_KEY=sk-xxxx

# 3. 启动（首次会自动用 jobs.json 构建向量索引，需联网调用嵌入 API，约 1~2 分钟）
python app.py

# 4. 浏览器打开
http://localhost:5000
```

> 首次启动做了什么：`app.py` 发现 `chroma_db` 为空时，会自动读取仓库内的 `jobs.json`（2000 条真实岗位）调用嵌入接口构建向量库，构建完成后才开始服务；之后再次启动会直接复用，秒开。

> 不要用 `file://` 双击打开 `index.html`——浏览器会拦截 `fetch` 请求，必须通过 `http://localhost:5000` 访问。

### 健康检查

```bash
curl http://localhost:5000/health
# -> {"status":"ok","has_key":true,"index_count":2000}
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
| GET | `/stats` | 城市 / 类型 / 学历 / 标签聚合统计（含 `by_edu`） |
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
# 检索召回评估（需 API key；对比「混合检索基线」与「+bge-reranker 重排」）
# 当前结果：命中率 hit@5 均为 100%，top5 平均相关性分 0.838 → 0.885（+5.6%）
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

## 已知局限与诚实声明

- **数据真实性**：库内 2000 条中仅约 222 条为公开渠道人工核校样本，其余约 1778 条为 AI 辅助生成（标注 `source="ai-augmented"`），用于补足样本量与问法覆盖，**不代表已核实的实时招聘信息，请勿据此直接投递**。
- **生成依赖外部 API**：检索与回答依赖硅基流动（DeepSeek / bge-m3）服务，密钥失效或受限流时会自动降级为「列出检索岗位」以保证有结果。
- **会话不上云**：多轮对话历史仅存于服务进程内存，重启即清空，仅用于本地演示。
- **CI 工作流**：`.github/workflows/eval.yml` 已启用；每次 push / PR 到 `main` 自动跑 `pytest`，并（在仓库配置 `SILICONFLOW_API_KEY` Secret 后）额外跑召回评估。

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
