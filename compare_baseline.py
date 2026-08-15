"""裸 LLM 直答 vs RAG 回答对照，产出 Markdown 报告供人工抽查（不做自动打分）。"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from config import CHAT_MODEL, EMBEDDING_MODEL, QUERY_REWRITE, RETRIEVAL_SCORE_THRESHOLD
from eval import EVAL_QUESTIONS_PATH, load_questions
from llm_providers import get_chat_llm, rewrite_query
from vector_stores import get_vector_store

BASELINE_PROMPT = ChatPromptTemplate.from_template(
    "你是 LangChain 开发助手。凭你已掌握的知识回答以下 LangChain 问题，"
    "涉及代码时给出一个可直接运行的最小示例。\n\n问题：{question}"
)

# 与 api.py 中的 PROMPT 保持一致
RAG_PROMPT = ChatPromptTemplate.from_template("""你是 LangChain 开发助手。严格依据用户消息中提供的官方文档片段回答问题：
1. 只使用文档片段中的 API 与写法；文档未涵盖的，明确回答"当前文档中未找到"，不要凭记忆补全。
2. 涉及代码时，给出一个可直接运行的最小示例。
3. 先给结论，再给依据，正文控制在 300 字以内（代码除外）。

检索到的文档上下文：
{context}

用户问题：{question}

回答：""")


def rag_answer(store, llm, question: str, k: int) -> tuple[str, list[str]]:
    query_text = rewrite_query(question) if QUERY_REWRITE else question
    results = store.similarity_search(query_text, k=k, score_threshold=RETRIEVAL_SCORE_THRESHOLD)
    context = "\n\n---\n\n".join(r["content"] for r in results) or "（未检索到相关文档）"
    message = llm.invoke(RAG_PROMPT.format(context=context, question=question))
    sources = list(dict.fromkeys(r["source"] for r in results))
    return str(message.content), sources


def main():
    parser = argparse.ArgumentParser(description="裸 LLM vs RAG 对照")
    parser.add_argument("--questions", type=Path, default=EVAL_QUESTIONS_PATH)
    parser.add_argument("--limit", type=int, default=8, help="取前 N 题（默认 8）")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/baseline_comparison.md"))
    args = parser.parse_args()

    questions = load_questions(args.questions)[: args.limit]
    print(f"{len(questions)} 道题，加载向量库与 LLM...")

    store = get_vector_store()
    llm, chat_provider = get_chat_llm(temperature=0)
    print(f"Chat: {chat_provider} | Embedding: {EMBEDDING_MODEL} | Store: {store.get_stats()['backend']}")

    lines = [
        "# 裸 LLM 直答 vs RAG 对照",
        "",
        f"- 生成时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Chat 模型: {chat_provider} | Embedding: {EMBEDDING_MODEL}",
        f"- 检索: top-{args.k}, 阈值 {RETRIEVAL_SCORE_THRESHOLD}, 查询改写 {'开' if QUERY_REWRITE else '关'}",
        "",
    ]
    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"[{i}/{len(questions)}] {question}")
        baseline = str(llm.invoke(BASELINE_PROMPT.format(question=question)).content)
        rag, sources = rag_answer(store, llm, question, k=args.k)
        lines += [
            f"## Q{i}: {question}",
            "",
            "**裸模型直答：**",
            "",
            baseline,
            "",
            f"**RAG 回答**（来源: {', '.join(sources) or '无'}）：",
            "",
            rag,
            "",
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n对照报告 → {args.output}")


if __name__ == "__main__":
    main()
