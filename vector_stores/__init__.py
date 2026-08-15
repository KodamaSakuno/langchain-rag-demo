from config import CHROMA_PERSIST_DIR, CHUNKS_PATH, PG_CONNECTION, VECTOR_STORE_BACKEND
from llm_providers import get_embeddings

from .chroma_store import ChromaVectorStore


def get_vector_store():
    if VECTOR_STORE_BACKEND == "pgvector":
        from .pg_store import PgVectorStore

        embeddings, _ = get_embeddings()
        return PgVectorStore(connection=PG_CONNECTION, embedding_function=embeddings)

    if VECTOR_STORE_BACKEND not in ("chroma", "hybrid"):
        raise ValueError(f"Unknown VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}. Use 'chroma', 'hybrid' or 'pgvector'")

    embeddings, _ = get_embeddings()
    store = ChromaVectorStore(
        persist_dir=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    if VECTOR_STORE_BACKEND == "hybrid":
        from .hybrid_store import HybridVectorStore
        return HybridVectorStore(store, str(CHUNKS_PATH))

    return store
