import argparse
import json
from pathlib import Path
from typing import Any

from config import CHUNKS_PATH, EMBEDDING_MODEL
from vector_stores import get_vector_store


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    path = Path(chunks_path)
    if not path.exists():
        print(f"{path} 不存在，请先运行: python3 ingest.py")
        return []

    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": chunk["source"],
        "header_path": chunk.get("header_path", ""),
        "has_code": bool(chunk.get("has_code", False)),
        "chunk_id": chunk["chunk_id"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="清空后重建")
    args = parser.parse_args()

    store = get_vector_store()
    stats_before = store.get_stats()
    print(f"向量存储: {stats_before['backend']}, 存量: {stats_before.get('total_chunks', 0)} 块")
    print(f"Embedding 模型: {EMBEDDING_MODEL}")

    chunks = load_chunks()
    if not chunks:
        return

    n_sources = len({c["source"] for c in chunks})
    print(f"{len(chunks)} 个分块，来自 {n_sources} 个文件 ({CHUNKS_PATH})")

    texts = [c["text"] for c in chunks]
    metadatas = [build_metadata(c) for c in chunks]

    if args.rebuild:
        print("清空旧索引...")
        store.clear()

    print(f"写入向量库（{len(texts)} 条，内部嵌入，首次运行需下载模型）...")
    store.add_documents(texts=texts, metadatas=metadatas)

    stats_after = store.get_stats()
    print(f"\n建库完成: {stats_after}")


if __name__ == "__main__":
    main()
