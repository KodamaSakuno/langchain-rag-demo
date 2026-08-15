import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ==================== 文档与分块 ====================
DOCS_DIR = Path("data/docs")
CHUNKS_PATH = Path("data/chunks.jsonl")
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_CHARS = 150

# ==================== Embedding ====================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# hf = 本地 HuggingFace（默认）；api = OpenAI 兼容接口
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hf").lower()
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://ai.gitee.com/v1")

# ==================== 聊天 LLM ====================
CHAT_API_KEY = os.getenv("CHAT_API_KEY")
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ==================== 检索 ====================
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "0") == "1"

# ==================== 向量存储 ====================
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
PG_CONNECTION = os.getenv("PG_CONNECTION", "postgresql+psycopg:///rag_demo")
# pgvector collection 名；对比不同 embedding 模型时可用独立 collection 隔离索引
PG_COLLECTION = os.getenv("PG_COLLECTION", "langchain_docs")

# ==================== 记忆 ====================
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory").lower()

# ==================== 部署形态 ====================
# 启动时预热向量库与嵌入模型（本地/EC2 需要，避免子线程首次推理竞态）；
# Lambda 冷启动场景设为 0 跳过，首次请求时惰性初始化
WARMUP = os.getenv("WARMUP", "1") == "1"
# pgvector 连接池：Lambda 单实例串行处理请求，池开大会打爆托管库连接数
PG_POOL_SIZE = int(os.getenv("PG_POOL_SIZE", "5"))

# 多 Agent 协作：开启后主 Agent 输出前会把回答草稿交给审校员 subagent 核对
MULTI_AGENT = os.getenv("MULTI_AGENT", "0") == "1"
