from config import CHROMA_PERSIST_DIR, CHUNKS_PATH, VECTOR_STORE_BACKEND
from llm_providers import get_embeddings

from .chroma_store import ChromaVectorStore


def get_vector_store():
    if VECTOR_STORE_BACKEND not in ("chroma", "hybrid"):
        raise ValueError(f"Unknown VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}. Use 'chroma' or 'hybrid'")

    embeddings, _ = get_embeddings()
    store = ChromaVectorStore(
        persist_dir=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    if VECTOR_STORE_BACKEND == "hybrid":
        from .hybrid_store import HybridVectorStore
        return HybridVectorStore(store, str(CHUNKS_PATH))

    return store
