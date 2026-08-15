from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from .base import BaseVectorStore


class PgVectorStore(BaseVectorStore):
    """Postgres + pgvector 后端。向量与记忆（PostgresSaver）可共用一个库，状态全外置。

    注意：distance 为 cosine distance，similarity = 1 - distance（即余弦相似度），
    与 chroma 后端的分数尺度不同，切换后端后需用 eval.py 重新校准阈值。
    """

    def __init__(self, connection: str, embedding_function: Embeddings, collection: str = "langchain_docs",
                 pool_size: int = 5):
        self.connection = connection
        self.collection = collection
        self.embedding_function = embedding_function
        # Lambda 等 serverless 场景实例并发为 1，传 pool_size=1 避免打爆托管库连接数
        self._engine = create_engine(connection, pool_size=pool_size, max_overflow=0)
        self._store = self._make_store()

    def _make_store(self) -> PGVector:
        return PGVector(
            embeddings=self.embedding_function,
            collection_name=self.collection,
            connection=self._engine,
            use_jsonb=True,
        )

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self._store.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query_text: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        docs_and_scores = self._store.similarity_search_with_relevance_scores(query=query_text, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "unknown"),
                "similarity": round(float(score), 4)
            }
            for doc, score in docs_and_scores
            if score >= score_threshold
        ]

    def clear(self) -> None:
        self._store.delete_collection()
        # delete_collection 后旧实例无法再写入（langchain-postgres 0.0.x），重建 collection
        self._store = self._make_store()

    def get_stats(self) -> Dict[str, Any]:
        with self._engine.connect() as conn:
            total = conn.execute(text("SELECT count(*) FROM langchain_pg_embedding")).scalar()
        return {
            "total_chunks": total or 0,
            "backend": "pgvector",
            "persist_dir": self.connection.rsplit("/", 1)[-1],  # 库名
        }
