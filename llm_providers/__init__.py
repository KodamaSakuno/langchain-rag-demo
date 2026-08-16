from langchain_openai import ChatOpenAI

from config import (
    CHAT_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BACKEND,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
)


def get_chat_llm(temperature: float = 0):
    if not CHAT_API_KEY:
        raise ValueError("CHAT_API_KEY not set")

    return (
        ChatOpenAI(
            model=CHAT_MODEL,
            api_key=CHAT_API_KEY,
            base_url=CHAT_BASE_URL,
            temperature=temperature,
            # 网络挂起防护：单次调用超时 120s、最多重试 2 次，
            # 否则一个 hang 住的请求会把整条链路（或评测进程）永久卡住
            request_timeout=120,
            max_retries=2,
        ),
        CHAT_MODEL,
    )


class _NormalizedEmbeddings:
    """L2 归一化包装：API 返回的向量未必归一化，而检索阈值语义依赖单位向量。

    附带小批量 + 重试：部分供应商（如 Gitee AI）对单请求 token 总量/频率有限制，
    报错是误导性的 400"token 计算失败"，稍等重试即可。
    """

    BATCH_SIZE = 16
    MAX_RETRIES = 6

    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def _norm(v):
        import math

        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    def embed_documents(self, texts):
        import time

        vecs = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            for attempt in range(self.MAX_RETRIES):
                try:
                    vecs.extend(self._inner.embed_documents(batch))
                    break
                except Exception:
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    time.sleep(min(2 ** attempt * 2, 30))
        return [self._norm(v) for v in vecs]

    def embed_query(self, text):
        return self._norm(self._inner.embed_query(text))


def get_embeddings():
    if EMBEDDING_BACKEND == "api":
        from langchain_openai import OpenAIEmbeddings

        if not EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_BACKEND=api 需要设置 EMBEDDING_API_KEY")
        return (
            _NormalizedEmbeddings(
                OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    api_key=EMBEDDING_API_KEY,
                    base_url=EMBEDDING_BASE_URL,
                    # 与 _NormalizedEmbeddings.BATCH_SIZE 对齐，避免内部再攒大批次
                    chunk_size=16,
                )
            ),
            f"{EMBEDDING_MODEL} (api)",
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    return (
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        ),
        EMBEDDING_MODEL,
    )


REWRITE_INSTRUCTION = (
    "把用户问题改写为用于检索 LangChain 英文技术文档的英文关键词查询。"
    "保留问题中的专有名词（如 checkpointer、middleware），只输出查询本身，不要解释。\n\n"
    "用户问题：{question}"
)


def rewrite_query(question: str) -> str:
    try:
        llm, _ = get_chat_llm(temperature=0)
        rewritten = str(llm.invoke(REWRITE_INSTRUCTION.format(question=question)).content).strip()
        return rewritten or question
    except Exception:
        return question

