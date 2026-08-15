import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from agent import build_agent
from config import EMBEDDING_MODEL, RETRIEVAL_SCORE_THRESHOLD

app = FastAPI(title="LangChain 文档 RAG 助手")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent, chat_provider, store = build_agent()
print(f"服务启动 | Chat: {chat_provider} | Embedding: {EMBEDDING_MODEL} | 模式: Agent")


def _extract_token_usage(messages: list) -> dict[str, Any]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        um = getattr(m, "usage_metadata", None) or {}
        if not um:
            continue
        usage["prompt_tokens"] += um.get("input_tokens") or 0
        usage["completion_tokens"] += um.get("output_tokens") or 0
        usage["total_tokens"] += um.get("total_tokens") or 0
    return usage


def _parse_agent_result(messages: list) -> dict[str, Any]:
    """从 Agent 的消息序列中提取答案、引用与工具调用记录。"""
    answer = ""
    results = []
    tool_calls = []
    for m in messages:
        if isinstance(m, AIMessage):
            if m.content:
                answer = str(m.content)
            for tc in getattr(m, "tool_calls", None) or []:
                tool_calls.append({"tool": tc["name"], "query": tc["args"].get("query", "")})
        elif isinstance(m, ToolMessage) and m.name == "search_docs":
            results.extend(getattr(m, "artifact", None) or [])

    sources = list(dict.fromkeys(r["source"] for r in results))
    citations = []
    seen = set()
    for r in results:
        key = (r["source"], r["metadata"].get("header_path", ""))
        if key not in seen:
            seen.add(key)
            citations.append({"source": key[0], "section": key[1], "score": r["similarity"]})
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "tool_calls": tool_calls,
    }


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    thread_id: str | None = Field(None, description="会话 ID，相同 ID 保持多轮上下文；不传则每次新建")


class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[str]
    citations: list[dict[str, Any]]
    tool_calls: list[dict[str, str]]
    model_info: dict[str, Any]
    token_usage: dict[str, Any]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    thread_id = req.thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    # checkpointer 会返回完整会话历史，记录调用前的消息数以便只解析本轮新增
    prev_len = len(agent.get_state(config).values.get("messages", []))
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": req.question}]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent 执行失败，请检查 CHAT_API_KEY 与网络：{e}")

    messages = result["messages"][prev_len:]
    parsed = _parse_agent_result(messages)
    return QueryResponse(
        **parsed,
        thread_id=thread_id,
        model_info={"chat": chat_provider, "embedding": EMBEDDING_MODEL},
        token_usage=_extract_token_usage(messages),
    )


@app.get("/")
def root():
    return {
        "message": "LangChain 文档 RAG 助手（Agent 模式）",
        "store": store.get_stats(),
        "models": {"chat": chat_provider, "embedding": EMBEDDING_MODEL},
        "retrieval": {"score_threshold": RETRIEVAL_SCORE_THRESHOLD}
    }


from fastapi.staticfiles import StaticFiles

app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
