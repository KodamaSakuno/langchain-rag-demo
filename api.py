import threading
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from agent import build_agent
from config import DEMO_ACCESS_CODE, EMBEDDING_BACKEND, EMBEDDING_BASE_URL, EMBEDDING_MODEL, RETRIEVAL_SCORE_THRESHOLD

app = FastAPI(title="LangChain 文档 RAG 助手")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 惰性构建：Lambda 冷启动时 /ui 等静态路由不应被 agent 构建（建库连接、checkpointer 建表）拖累；
# 首个需要 agent/store 的请求才触发构建。加锁防并发请求重复构建（Lambda 预留并发 >1 或 uvicorn 线程池）。
_runtime = None
_runtime_lock = threading.Lock()


def get_runtime():
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = build_agent()
                print(f"Agent 已构建 | Chat: {_runtime[2]} | Embedding: {EMBEDDING_MODEL} ({EMBEDDING_BACKEND})")
    return _runtime


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
    review = ""
    for m in messages:
        if isinstance(m, AIMessage):
            if m.content:
                answer = str(m.content)
            for tc in getattr(m, "tool_calls", None) or []:
                tool_calls.append({"tool": tc["name"], "query": tc["args"].get("query") or tc["args"].get("question", "")})
        elif isinstance(m, ToolMessage) and m.name == "search_docs":
            results.extend(getattr(m, "artifact", None) or [])
        elif isinstance(m, ToolMessage) and m.name == "review_answer":
            review = str(m.content)

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
        "review": review,
    }


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    thread_id: str | None = Field(None, description="会话 ID，相同 ID 保持多轮上下文；不传则每次新建")
    multi_agent: bool | None = Field(None, description="是否启用审校员 subagent；不传或 false 为关")
    access_code: str | None = Field(None, description="访问码；服务端设置了 DEMO_ACCESS_CODE 时必传")


def _check_access(code: str | None):
    if DEMO_ACCESS_CODE and code != DEMO_ACCESS_CODE:
        raise HTTPException(status_code=403, detail="需要有效的访问码（demo 防滥用）")


class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[str]
    citations: list[dict[str, Any]]
    tool_calls: list[dict[str, str]]
    review: str = ""   # 多 Agent 模式下审校员的结论；未开启时为空
    model_info: dict[str, Any]
    token_usage: dict[str, Any]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    _check_access(req.access_code)
    agent, agent_reviewed, chat_provider, _store = get_runtime()
    thread_id = req.thread_id or uuid.uuid4().hex
    use_multi_agent = bool(req.multi_agent)  # None/False 都视为关，由前端复选框显式开启
    selected = agent_reviewed if use_multi_agent else agent
    config = {"configurable": {"thread_id": thread_id}}
    # checkpointer 会返回完整会话历史，记录调用前的消息数以便只解析本轮新增
    prev_len = len(selected.get_state(config).values.get("messages", []))
    try:
        result = selected.invoke(
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
        model_info={"chat": chat_provider, "embedding": EMBEDDING_MODEL, "multi_agent": use_multi_agent},
        token_usage=_extract_token_usage(messages),
    )


@app.get("/")
def root():
    _agent, _agent_reviewed, chat_provider, store = get_runtime()
    return {
        "message": "LangChain 文档 RAG 助手（Agent 模式）",
        "store": store.get_stats(),
        "models": {"chat": chat_provider, "embedding": f"{EMBEDDING_MODEL} ({EMBEDDING_BACKEND})"},
        "retrieval": {"score_threshold": RETRIEVAL_SCORE_THRESHOLD},
        "access_gate": bool(DEMO_ACCESS_CODE),  # 前端据此决定是否提示输入访问码
    }


@app.get("/debug/embedding", include_in_schema=False)
def debug_embedding(code: str | None = None):
    """部署自检：真实调用一次 embedding 接口，验证网络可达与配置正确。

    排查部署环境问题用（如 Lambda 上怀疑连不上 embedding API 时），
    一条 curl 即可定位，不用等业务接口报错。
    """
    import time

    _check_access(code)

    _agent, _agent_reviewed, _chat_provider, store = get_runtime()

    # hybrid store 包了一层，从内部的向量 store 取 embedding 函数
    emb = getattr(store, "embedding_function", None)
    if emb is None:
        emb = getattr(getattr(store, "_vector", None), "embedding_function", None)
    if emb is None:
        raise HTTPException(status_code=500, detail="无法从向量存储获取 embedding 函数")

    start = time.monotonic()
    try:
        vec = emb.embed_query("ping")
    except Exception as e:
        elapsed = round(time.monotonic() - start, 2)
        raise HTTPException(
            status_code=502,
            detail=f"embedding 调用失败（{elapsed}s）：{type(e).__name__}: {e}",
        )
    return {
        "backend": EMBEDDING_BACKEND,
        "model": EMBEDDING_MODEL,
        "base_url": EMBEDDING_BASE_URL if EMBEDDING_BACKEND == "api" else None,
        "dim": len(vec),
        "elapsed_s": round(time.monotonic() - start, 2),
    }


# 前端单页（同源部署时访问 /ui）。不用 StaticFiles 挂载：Mangum/Lambda 下
# StaticFiles 的目录斜杠重定向会因 scope 路径重构问题形成 307 循环，
# 前端是单文件，显式 FileResponse 路由在任何 ASGI 适配层下行为都确定
from fastapi.responses import FileResponse


@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
def ui():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
