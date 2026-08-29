"""构建/重建岗位向量库：读 jobs.json -> 硅基流动 bge-m3 嵌入 -> Chroma 持久化。
逻辑已抽到 rag_core.build_collection，本文件仅作命令行入口。
"""
from rag_core import build_collection


def main():
    build_collection()


if __name__ == "__main__":
    main()
