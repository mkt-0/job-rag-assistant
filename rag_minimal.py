# ===== 极简 RAG 陪练项目（逐段对应 Python 第1周基础）=====
# RAG = 检索增强生成：文档切块 → 向量化存库 → 提问时检索相关块 → 交给大模型回答
# 安装：pip install chromadb openai
# API：硅基流动 https://cloud.siliconflow.cn 注册拿 key（新用户有免费额度）

# ---------- 1. 配置区：变量 + 字符串 ----------
import os
# 读取 API key：优先项目根目录 .env，其次环境变量。
# ⚠️ 请勿把真实 key 写进代码——一旦提交到公开仓库就会泄露。
def _load_key():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("SILICONFLOW_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return os.getenv("SILICONFLOW_API_KEY", "")
API_KEY = _load_key()
DOC_PATH = "knowledge.txt"            # 你的文档路径
LLM_MODEL = "deepseek-ai/DeepSeek-V3" # 硅基流动上的模型名（字符串）


# ---------- 2. 读取并切分文档：文件读写 + 列表 + for循环 ----------
def load_and_split(path, chunk_size=80):
    with open(path, "r", encoding="utf-8") as f:  # 读文件
        text = f.read()
    chunks = []                                   # 空列表，装切好的块
    for i in range(0, len(text), chunk_size):      # 循环，步长=块大小
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)                      # 加进列表
    return chunks


# ---------- 3. 建库：向量数据库（Chroma 本地运行） ----------
def build_db(chunks):
    import chromadb
    client = chromadb.Client()                    # 本地客户端
    collection = client.create_collection("my_doc")  # 用默认本地嵌入（首次下载模型）
    for idx, c in enumerate(chunks):              # enumerate 同时拿到序号和内容
        collection.add(documents=[c], ids=[f"chunk_{idx}"])  # f-string 拼 id
    return collection


# ---------- 4. 检索：相似度查询（函数 + 字典取值） ----------
def retrieve(collection, question, n=3):
    res = collection.query(query_texts=[question], n_results=n)
    return res["documents"][0]                    # 返回最相关的 n 块


# ---------- 5. 组装并问大模型：f-string + API 调用 ----------
def ask(collection, question, api_key):
    ctx = retrieve(collection, question)          # 先检索
    context = "\n".join(ctx)                      # 把相关块拼成一段
    prompt = f"""根据下面资料回答问题，只根据资料回答：
资料：
{context}
问题：{question}"""                               # f-string 嵌入变量
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# ---------- 主程序 ----------
if __name__ == "__main__":
    chunks = load_and_split(DOC_PATH)
    print(f"文档切成 {len(chunks)} 块")           # f-string + len
    col = build_db(chunks)
    print("检索演示（问'什么是列表'）：")
    for c in retrieve(col, "什么是列表"):
        print(" -", c)
    if API_KEY and not API_KEY.startswith("在此填写"):        # 填了真实 key 才跑 LLM
        print("\nLLM 回答：")
        print(ask(col, "Python 字典怎么用", API_KEY))
    else:
        print("\n⚠️ 还没填 API_KEY（在 .env 或环境变量中配置后重跑），LLM 问答部分待你填 key 后运行。")
