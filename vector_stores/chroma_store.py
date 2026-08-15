import os
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .base import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_dir: str, embedding_function: Embeddings, collection: str = "langchain_docs"):
        self.persist_dir = persist_dir
        self.collection = collection
        self.embedding_function = embedding_function
        self._store = None

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_function,
                collection_name=self.collection
            )
        return self._store

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self.store.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        docs = self.store.similarity_search(query=query_text, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "unknown"),
                "similarity": None
            }
            for doc in docs
        ]

    def clear(self) -> None:
        self.store.delete_collection()
        self._store = None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.store._collection.count(),
            "backend": "chroma",
            "persist_dir": os.path.abspath(self.persist_dir)
        }
