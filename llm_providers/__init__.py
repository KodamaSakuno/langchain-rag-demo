from langchain_openai import ChatOpenAI

from config import CHAT_API_KEY, CHAT_BASE_URL, CHAT_MODEL, EMBEDDING_MODEL


def get_chat_llm(temperature: float = 0):
    if not CHAT_API_KEY:
        raise ValueError("CHAT_API_KEY not set")

    return (
        ChatOpenAI(
            model=CHAT_MODEL,
            api_key=CHAT_API_KEY,
            base_url=CHAT_BASE_URL,
            temperature=temperature,
        ),
        CHAT_MODEL,
    )


def get_embeddings():
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

