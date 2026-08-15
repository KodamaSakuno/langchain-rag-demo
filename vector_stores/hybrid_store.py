import json
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseVectorStore
from .bm25 import BM25, tokenize

RRF_K = 60


class HybridVectorStore(BaseVectorStore):
    def __init__(self, vector_store: BaseVectorStore, chunks_path: str):
        self._vector = vector_store
        self._chunks: Dict[str, dict] = {}
        with Path(chunks_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    self._chunks[chunk["chunk_id"]] = chunk
        self._ids = list(self._chunks)
        self._bm25 = BM25([tokenize(self._chunks[cid]["text"]) for cid in self._ids])

    def similarity_search(self, query_text: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        vec_results = self._vector.similarity_search(query_text, k=k, score_threshold=score_threshold)
        vec_scores = {}
        rrf = {}

        for rank, r in enumerate(vec_results):
            cid = r["metadata"].get("chunk_id")
            if cid is None:
                continue
            vec_scores[cid] = r["similarity"]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, (idx, _) in enumerate(self._bm25.search(query_text, k=k)):
            cid = self._ids[idx]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        top_ids = sorted(rrf, key=lambda cid: -rrf[cid])[:k]
        results = []
        for cid in top_ids:
            chunk = self._chunks[cid]
            results.append({
                "content": chunk["text"],
                "metadata": {
                    "source": chunk["source"],
                    "header_path": chunk.get("header_path", ""),
                    "has_code": bool(chunk.get("has_code", False)),
                    "chunk_id": cid,
                },
                "source": chunk["source"],
                "similarity": vec_scores.get(cid),
            })
        return results

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self._vector.add_documents(texts, metadatas)

    def clear(self) -> None:
        self._vector.clear()

    def get_stats(self) -> Dict[str, Any]:
        stats = self._vector.get_stats()
        stats["backend"] = f"hybrid({stats['backend']}+bm25)"
        return stats
