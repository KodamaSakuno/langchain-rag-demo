"""带溯源的评测题生成器：从语料 chunk 出候选题，来源标注在生成时天然正确。

四类题型（默认 16/8/8/4 = 36 题）：
  - direct_hit  直接命中：单 chunk 可答，测检索+生成基本盘
  - cross_doc   跨文档综合：需同主题两个文档的 chunk 综合作答
  - api_trap    时效陷阱：凭训练记忆会答旧 API（AgentExecutor 等），正确答案以当前文档为准
  - oob         超纲拒答：与语料主题相邻但未被覆盖，期望明确回答"未找到"（用向量库验证确未覆盖）

铁律：参考答案只能依据喂给 LLM 的 chunk 原文撰写，prompt 中明确禁止凭记忆补全。

输出 data/eval_questions_draft.jsonl，字段比正式题库多 reference_answer / category /
chunk_ids / needs_review（api_trap 与 oob 需人工核对），核对后转正式题库。

用法：python3 gen_questions.py [--seed 42] [--output data/eval_questions_draft.jsonl]
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from config import CHUNKS_PATH, RETRIEVAL_SCORE_THRESHOLD
from llm_providers import get_chat_llm

QUOTAS = {"direct_hit": 16, "cross_doc": 8, "api_trap": 8, "oob": 4}

# 同主题文档组：跨文档综合题从同一组内取两个不同来源，保证"综合"是自然的
TOPIC_GROUPS = [
    ["multi-agent/subagents.mdx", "multi-agent/router.mdx", "multi-agent/handoffs.mdx",
     "multi-agent/custom-workflow.mdx", "multi-agent/skills.mdx"],
    ["middleware/built-in.mdx", "middleware/custom.mdx", "middleware/overview.mdx"],
    ["short-term-memory.mdx", "long-term-memory.mdx"],
    ["streaming.mdx", "event-streaming.mdx"],
    ["models.mdx", "messages.mdx"],
    ["tools.mdx", "structured-output.mdx"],
    ["knowledge-base.mdx", "deepagents-retrieval.mdx"],
    ["sql-agent.mdx", "multi-agent/skills-sql-assistant.mdx"],
    ["agents.mdx", "context-engineering.mdx"],
]

# 1.x 新 API 标记：含这些的 chunk 才有时效陷阱可挖
TRAP_MARKERS = ["create_agent", "checkpointer", "middleware", "@tool", "init_chat_model",
                "InMemorySaver", "ToolStrategy", "ProviderStrategy", "ToolNode", "load_skill"]

_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)


def ask_json(llm, prompt: str):
    """让 LLM 输出 JSON 并解析；容忍 markdown 代码围栏。"""
    raw = str(llm.invoke(prompt).content).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
    m = _JSON_RE.search(raw)
    return json.loads(m.group(0) if m else raw)


PROMPT_DIRECT = """你是 LangChain 文档题库出题人。基于以下文档片段出一道中文技术问答题。
要求：
1. 问题必须能仅凭该片段完整回答，不依赖片段外的知识
2. 问题要像开发者真实提问，不出现"根据文档""文中""该片段"等字样
3. 参考答案只能依据片段内容撰写，严禁使用片段之外的知识；150 字以内
4. 提取 2~4 个关键要点：任何正确答案都必须出现的 API 名/专有名词，用原文拼写（如 create_agent、thread_id）。要点必须是能做子串匹配的字面量，不带括号省略号等修饰（写 .implement 而非 .implement(...)）
严格输出 JSON：{{"question": "...", "answer": "...", "points": ["...", "..."]}}

文档片段（来源 {source}，章节 {header}）：
{chunk}"""

PROMPT_CROSS = """你是 LangChain 文档题库出题人。基于以下两个不同文档的片段，出一道需要综合两者才能完整回答的中文技术问答题（对比、组合使用、场景区分等）。
要求：
1. 问题必须同时依赖两个片段，单独一个片段答不全
2. 问题要像开发者真实提问，不出现"根据文档""文中"等字样
3. 参考答案只能依据两个片段撰写，严禁使用之外的知识；200 字以内
4. 提取 2~4 个关键要点：任何正确答案都必须出现的 API 名/专有名词，用原文拼写；要点必须是字面量，不带括号省略号等修饰
严格输出 JSON：{{"question": "...", "answer": "...", "points": ["...", "..."]}}

片段 A（来源 {source_a}，章节 {header_a}）：
{chunk_a}

片段 B（来源 {source_b}，章节 {header_b}）：
{chunk_b}"""

PROMPT_TRAP = """以下是 LangChain 1.x 当前文档片段。请出一道"API 时效陷阱"题：
凭过时训练记忆的模型会用旧版 API 回答（如 AgentExecutor、create_react_agent、
RunnableWithMessageHistory、ConversationBufferMemory、initialize_agent 等），
而按当前文档应使用新 API。
要求：
1. 问题像开发者真实提问，中文，不提示"新版/旧版"
2. 参考答案严格依据片段给出当前正确做法；150 字以内
3. stale_answer 字段：凭过时记忆会给出的典型错误答案（一句话，供人工核对题目是否有区分度）
4. 提取 2~4 个关键要点：正确答案必须出现的新 API 名，用原文拼写；要点必须是字面量，不带括号省略号等修饰
严格输出 JSON：{{"question": "...", "answer": "...", "stale_answer": "...", "points": ["...", "..."]}}

文档片段（来源 {source}，章节 {header}）：
{chunk}"""

PROMPT_OOB = """一个 LangChain 文档问答系统的语料只覆盖以下文档：
{sources}

请列 {n} 个"看似能答、实则超纲"的中文问题候选：与 LangChain 生态相关，但不在上述文档覆盖范围内
（如云平台部署、其他语言版本、竞品对比集成、定价、未列出的组件等）。不要与这些已有超纲题重复：
LangGraph Platform 部署灰度发布 / LlamaIndex 集成 / Go 语言版本。
严格输出 JSON 数组：["...", "..."]"""


def load_chunks() -> list[dict]:
    chunks = [json.loads(l) for l in Path(CHUNKS_PATH).open(encoding="utf-8") if l.strip()]
    return [c for c in chunks if len(c["text"]) >= 400]  # 太短的分块信息量不足出题


def gen_direct(llm, chunks, rng, n):
    by_source = defaultdict(list)
    for c in chunks:
        by_source[c["source"]].append(c)
    sources = [s for s, cs in by_source.items() if len(cs) >= 8]
    rng.shuffle(sources)
    out = []
    for s in sources[:n]:
        c = rng.choice(by_source[s])
        r = ask_json(llm, PROMPT_DIRECT.format(source=s, header=c["header_path"], chunk=c["text"]))
        out.append({"question": r["question"], "category": "direct_hit",
                    "expected_sources": [s], "expected_points": r["points"],
                    "reference_answer": r["answer"], "chunk_ids": [c["chunk_id"]],
                    "expect_not_found": False, "needs_review": False})
        print(f"  [direct_hit] {r['question'][:40]}", flush=True)
    return out


def gen_cross(llm, chunks, rng, n):
    by_source = defaultdict(list)
    for c in chunks:
        by_source[c["source"]].append(c)
    groups = [g for g in TOPIC_GROUPS if sum(s in by_source for s in g) >= 2]
    rng.shuffle(groups)
    out = []
    for g in groups[:n]:
        sa, sb = rng.sample([s for s in g if s in by_source], 2)
        ca, cb = rng.choice(by_source[sa]), rng.choice(by_source[sb])
        r = ask_json(llm, PROMPT_CROSS.format(
            source_a=sa, header_a=ca["header_path"], chunk_a=ca["text"],
            source_b=sb, header_b=cb["header_path"], chunk_b=cb["text"]))
        out.append({"question": r["question"], "category": "cross_doc",
                    "expected_sources": [sa, sb], "expected_points": r["points"],
                    "reference_answer": r["answer"], "chunk_ids": [ca["chunk_id"], cb["chunk_id"]],
                    "expect_not_found": False, "needs_review": False})
        print(f"  [cross_doc] {r['question'][:40]}", flush=True)
    return out


def gen_trap(llm, chunks, rng, n):
    pool = [c for c in chunks if any(m in c["text"] for m in TRAP_MARKERS)]
    rng.shuffle(pool)
    out, used = [], set()
    for c in pool:
        if len(out) >= n:
            break
        if c["source"] in used:
            continue  # 每来源最多一题，保证覆盖
        r = ask_json(llm, PROMPT_TRAP.format(source=c["source"], header=c["header_path"], chunk=c["text"]))
        used.add(c["source"])
        out.append({"question": r["question"], "category": "api_trap",
                    "expected_sources": [c["source"]], "expected_points": r["points"],
                    "reference_answer": r["answer"], "stale_answer": r.get("stale_answer", ""),
                    "chunk_ids": [c["chunk_id"]],
                    "expect_not_found": False, "needs_review": True})
        print(f"  [api_trap] {r['question'][:40]}", flush=True)
    return out


def gen_oob(llm, chunks, rng, n):
    """超纲题：LLM 提名候选，再用向量库验证确实未被语料覆盖（检索无达阈值结果）。"""
    from vector_stores import get_vector_store

    sources = sorted({c["source"] for c in chunks})
    candidates = ask_json(llm, PROMPT_OOB.format(
        sources="\n".join(sources), n=n * 2))  # 多要一倍，验证会刷掉一部分
    store = get_vector_store()
    out = []
    for q in candidates:
        if len(out) >= n:
            break
        hits = store.similarity_search(q, k=3, score_threshold=RETRIEVAL_SCORE_THRESHOLD)
        if hits:
            print(f"  [oob 跳过] 语料内可检索到：{q[:40]} → {[h['source'] for h in hits]}", flush=True)
            continue
        out.append({"question": q, "category": "oob", "expected_sources": [],
                    "expected_points": [], "reference_answer": "当前文档中未找到，明确拒答。",
                    "chunk_ids": [], "expect_not_found": True, "needs_review": True})
        print(f"  [oob] {q[:40]}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="带溯源的评测题生成器")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/eval_questions_draft.jsonl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    chunks = load_chunks()
    llm, provider = get_chat_llm(temperature=0)
    print(f"{len(chunks)} 个可用 chunk | Chat: {provider} | 目标配比 {QUOTAS}", flush=True)

    questions = (
        gen_direct(llm, chunks, rng, QUOTAS["direct_hit"])
        + gen_cross(llm, chunks, rng, QUOTAS["cross_doc"])
        + gen_trap(llm, chunks, rng, QUOTAS["api_trap"])
        + gen_oob(llm, chunks, rng, QUOTAS["oob"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"\n已生成 {len(questions)} 题 → {args.output} | {dict(Counter(q['category'] for q in questions))}")
    print("needs_review=True 的题（api_trap / oob）请人工核对后再转正式题库")


if __name__ == "__main__":
    main()
