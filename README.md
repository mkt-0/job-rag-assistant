# 岗位 RAG · 实习/校招情报助手

基于真实招聘信息的检索增强问答（RAG）Web 应用，面向**大学生实习/校招求职**场景。用自然语言提问，即可从岗位库中检索相关 JD，并由大模型生成结构化、可落地的建议。

## 功能

- **自然语言问答**：如「无锡有哪些接受本科、薪资 15k 以上的 AI 岗？」「我零基础、大数据专业本科，能投哪些起步岗？」
- **可溯源防幻觉**：回答 grounding 在检索到的真实岗位资料上，并展示来源岗位卡片（公司 / 岗位 / 城市 / 薪资 / 类型）。
- **自助扩充数据**：界面「粘贴补充一条岗位」可把 JD 即时入库，立即可被检索；也可批量编辑 `jobs.json` 后重建索引。

## 技术栈

- 后端：Flask + Chroma 向量库 + 硅基流动（BAAI/bge-m3 中文嵌入、deepseek-ai/DeepSeek-V3 问答）
- 前端：单页 HTML（原生 JS，零构建）
- 数据：`jobs.json`（78 条可溯源的长三角岗位）

## 本地运行

```bash
cd Desktop/RAG
pip install -r requirements.txt
cp .env.example .env          # 然后填入你的硅基流动 API key
python build_index.py         # 首次构建向量库（生成 chroma_db/）
python app.py                 # 启动，访问 http://localhost:5000
```

> 服务监听 `0.0.0.0:5000`，同一局域网内其他设备可用本机 IP 访问。

## 目录结构

```
app.py            Flask 服务：/ 界面、/ask 问答、/jobs 列表、/add 补充
build_index.py    读取 jobs.json 构建 Chroma 向量索引
index.html        交互界面
jobs.json         岗位数据（可被 /add 或手工扩充）
knowledge.txt     早期单文档 RAG 示例（可忽略）
rag_minimal.py    最小化 RAG 模板（学习用）
```

## 数据说明

- `jobs.json` 为公开网络采集的**可溯源**岗位（截至 2026-08），并非全部实时在招；实际投递请以各招聘平台为准。
- 目标扩充到 1000 条：通过界面「粘贴补充一条岗位」，或把多个 JD 整理进 `jobs.json` 后重跑 `build_index.py` 即可。

## 安全

- API key 通过 `.env` 注入（已加入 `.gitignore`），请勿将含 key 的文件提交到任何仓库。
- 若 key 曾出现在聊天/日志中，建议前往 https://cloud.siliconflow.cn 重置。

## 许可证

MIT（示例项目，数据请遵守来源平台使用条款）
