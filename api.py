from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import EMBEDDING_MODEL, QUERY_REWRITE, RETRIEVAL_SCORE_THRESHOLD
from llm_providers import get_chat_llm, rewrite_query
from vector_stores import get_vector_store

app = FastAPI(title="LangChain 文档 RAG 助手")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = get_vector_store()
llm, chat_provider = get_chat_llm(temperature=0)

store_backend = store.get_stats()["backend"]
print(f"服务启动 | Chat: {chat_provider} | Embedding: {EMBEDDING_MODEL} | Store: {store_backend}")

PROMPT = ChatPromptTemplate.from_template("""你是 LangChain 开发助手。严格依据用户消息中提供的官方文档片段回答问题：
1. 只使用文档片段中的 API 与写法；文档未涵盖的，明确回答"当前文档中未找到"，不要凭记忆补全。
2. 涉及代码时，给出一个可直接运行的最小示例。
3. 先给结论，再给依据，正文控制在 300 字以内（代码除外）。

检索到的文档上下文：
{context}

用户问题：{question}

回答：""")

chain = PROMPT | llm


def _extract_token_usage(message) -> dict[str, Any]:
    um = getattr(message, "usage_metadata", None) or {}
    ru = (getattr(message, "response_metadata", None) or {}).get("token_usage", {}) or {}

    def pick(*candidates):
        for c in candidates:
            if c is not None:
                return c
        return None

    return {
        "prompt_tokens": pick(um.get("input_tokens"), ru.get("prompt_tokens")),
        "completion_tokens": pick(um.get("output_tokens"), ru.get("completion_tokens")),
        "total_tokens": pick(um.get("total_tokens"), ru.get("total_tokens")),
    }


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="检索返回的上下文块数")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    citations: list[dict[str, Any]]
    index_stats: dict[str, Any]
    model_info: dict[str, Any]
    token_usage: dict[str, Any]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    stats = store.get_stats()
    if stats.get("total_chunks", 0) == 0:
        return QueryResponse(
            answer="索引为空。请先运行：python3 ingest.py && python3 indexer.py",
            sources=[],
            citations=[],
            index_stats=stats,
            model_info={"chat": chat_provider, "embedding": EMBEDDING_MODEL},
            token_usage={}
        )

    try:
        query_text = rewrite_query(req.question) if QUERY_REWRITE else req.question
        results = store.similarity_search(query_text, k=req.top_k, score_threshold=RETRIEVAL_SCORE_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"文档检索失败：{e}")

    if not results:
        return QueryResponse(
            answer="未检索到相关文档。",
            sources=[],
            citations=[],
            index_stats=stats,
            model_info={"chat": chat_provider, "embedding": EMBEDDING_MODEL},
            token_usage={}
        )

    context_text = "\n\n---\n\n".join(r["content"] for r in results)
    sources = list(dict.fromkeys(r["source"] for r in results))
    citations = []
    seen = set()
    for r in results:
        key = (r["source"], r["metadata"].get("header_path", ""))
        if key not in seen:
            seen.add(key)
            citations.append({"source": key[0], "section": key[1], "score": r["similarity"]})

    try:
        message = chain.invoke({"context": context_text, "question": req.question})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM 生成失败，请检查 CHAT_API_KEY 与网络：{e}")

    return QueryResponse(
        answer=str(message.content),
        sources=sources,
        citations=citations,
        index_stats=stats,
        model_info={"chat": chat_provider, "embedding": EMBEDDING_MODEL},
        token_usage=_extract_token_usage(message),
    )


@app.get("/")
def root():
    return {
        "message": "LangChain 文档 RAG 助手",
        "store": store.get_stats(),
        "models": {"chat": chat_provider, "embedding": EMBEDDING_MODEL}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
