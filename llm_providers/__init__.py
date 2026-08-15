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
