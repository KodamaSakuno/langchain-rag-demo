from config import CHROMA_PERSIST_DIR, VECTOR_STORE_BACKEND
from llm_providers import get_embeddings

from .chroma_store import ChromaVectorStore


def get_vector_store():
    if VECTOR_STORE_BACKEND == "chroma":
        embeddings, _ = get_embeddings()
        return ChromaVectorStore(
            persist_dir=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

    else:
        raise ValueError(f"Unknown VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}. Use 'chroma'")
