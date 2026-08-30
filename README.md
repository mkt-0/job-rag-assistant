# 岗位 RAG 智能助手（job-rag-assistant）

面向大学生的**实习 / 校招 / 社招**岗位检索增强（RAG）问答系统。基于真实公开招聘数据，用向量检索 + 大语言模型，回答「某城市某方向的岗位要求 / 匹配度 / 投递建议」等问题。

> 技术栈：Flask + Chroma 向量库（硅基流动 `BAAI/bge-m3` 嵌入）+ 硅基流动 `DeepSeek-V3 / R1` 生成。
> 数据：长三角（无锡 / 苏州 / 南京 / 上海 / 常州 / 南通 / 杭州 / 宁波）真实在招岗位，当前 **222 条**，每条标注来源（猎聘 / 实习僧 / 智联 / 应届生求职网 / 高校就业网等）。

---

## 功能特性

- **真·接入大模型**：检索到的岗位资料作为上下文喂给 LLM，生成结构化、可行动的答案（不是简单关键词匹配）。
- **多轮对话**：`/chat` 与 `/chat_stream` 维护会话上下文，可追问、对比、深挖。
- **流式输出**：SSE 流式打字效果，回答逐字出现。
- **模型可切换**：标准 `DeepSeek-V3`（快） / 深度思考 `DeepSeek-R1`（慢但推理更强）。
- **智能检索**：可选「先让 LLM 把问题改写成检索关键词，再向量检索」，提升模糊问题的命中率。
- **数据看板**：页面顶部实时展示岗位总数、城市 / 类型 / 方向分布。
- **自助扩充**：`/add` 接口粘贴 JD 即可入库；`/rebuild` 一键重建索引。
- **质量保障**：`evaluation.py` 测检索召回，`tests/` 下 pytest 冒烟测试。

---

## 目录结构

```
.
├── app.py            # Flask 服务：/ask /ask_stream /chat /chat_stream /stats /jobs /add /rebuild
├── rag_core.py       # 共享核心：配置、客户端、嵌入、检索、LLM 对话/流式、城市归一化
├── build_index.py    # 用 jobs.json 重建 Chroma 向量索引
├── index.html        # 单页交互界面（流式 + 多轮 + 模型切换 + 看板）
├── jobs.json         # 岗位数据（222 条，字段规范，含 source 标注）
├── evaluation.py     # 检索召回@5 评估（标注问题集）
├── tests/test_api.py # pytest 冒烟测试（接口结构 + 无 key 报错路径）
├── requirements.txt
├── .env.example      # 环境变量模板
└── README.md
```

---

## 快速开始

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

---

## 数据从哪来

- 来源：猎聘、实习僧、智联校园、应届生求职网、BOSS 直聘聚合、全职招聘网、得早学就创、牛客、高校就业服务平台、企业官网等**公开渠道**检索整理。
- 诚实标注：每条岗位 `source` 字段记录检索来源；`updated` 记录整理月份。
- 规模与边界：当前 222 条为人工核校过的真实在招样本，覆盖长三角主要城市与主流技术方向（大数据 / 算法 / AI / 前端 / 测试 / 嵌入式 / 数据分析 / 产品 等）。**未做伪造填充**；要扩到更大规模建议接实时招聘 API 或写采集脚本（见下）。

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
# 检索召回评估（需 API key，离线嵌入）
python evaluation.py

# 接口冒烟测试（无需 key，测结构 + 无 key 报错路径）
pytest -q
```

---

## 安全

- 密钥只放在本地 `.env`，**不要提交**（已在 `.gitignore` 排除 `.env` 与 `chroma_db/`）。
- 多轮会话历史保存在进程内存，重启即清空，仅用于演示。
- 生成答案严格基于检索资料，缺失时如实说明「资料中未收录」，避免编造公司与薪资。

---

## 后续可扩展

- 接实时招聘 API（猎聘 / BOSS 开放能力或第三方）把样本扩到千级。
- 写 `collect_*.py` 自动采集脚本，替代手工整理。
- 加用户反馈标注，用评估集持续监控召回与答案质量。
