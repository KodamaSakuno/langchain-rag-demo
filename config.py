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

# ==================== 记忆 ====================
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory").lower()
