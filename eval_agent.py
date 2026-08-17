"""Agent 级效果评测：对比单 Agent 与多 Agent（规划员+审校员）在完整问答链路上的表现。

与 eval.py（检索层，直接测 similarity_search）不同，本脚本走真实 /query 链路：
Agent 自主检索、规划、审校。指标：
  - hit: citations 来源是否命中期望来源（客观）
  - honest: 文档未涵盖时是否明确回答"未找到"（诚实度）
  - 多 Agent 组额外统计审校结论分布

用法：
  WARMUP=0 VECTOR_STORE_BACKEND=pgvector EMBEDDING_BACKEND=api python3 eval_agent.py
"""
import argparse
import json
import time
import uuid
from pathlib import Path

from eval import EVAL_QUESTIONS_PATH, load_questions

from api import QueryRequest, query  # 复用真实端点逻辑（含双 Agent 构建）


def run_group(questions: list[dict], multi_agent: bool) -> dict:
    details = []
    hits = honest = 0
    total_latency = total_tokens = 0
    verdicts = {"通过": 0, "存疑": 0}

    for q in questions:
        req = QueryRequest(
            question=q["question"],
            thread_id=uuid.uuid4().hex,  # 每题独立会话，避免上下文污染
            multi_agent=multi_agent,
        )
        t0 = time.time()
        try:
            resp = query(req)
        except Exception as e:
            # 单题失败（如 LLM 超时）记录后继续，不拖垮整组
            latency = time.time() - t0
            details.append({
                "question": q["question"], "expected_sources": q["expected_sources"],
                "sources": [], "hit": False, "not_found": False, "review": "",
                "latency_s": round(latency, 1), "tokens": 0, "tool_calls": [],
                "error": str(e)[:200],
            })
            print(f"  ✗ {q['question'][:30]} | {latency:.0f}s | ERROR: {str(e)[:80]}")
            continue
        latency = time.time() - t0

        expected = set(q["expected_sources"])
        not_found = "未找到" in resp.answer
        if q.get("expect_not_found"):
            # 语料外诚实度题：正确行为是明确回答"未找到"，而非命中来源
            hit = not_found
            honest += hit
        else:
            hit = bool(expected & set(resp.sources))
            honest += not not_found
        hits += hit
        total_latency += latency
        total_tokens += resp.token_usage["total_tokens"]

        verdict = ""
        if resp.review:
            verdict = resp.review.splitlines()[0].strip()
            if verdict in verdicts:
                verdicts[verdict] += 1

        details.append({
            "question": q["question"],
            "expected_sources": q["expected_sources"],
            "sources": resp.sources,
            "hit": hit,
            "expect_not_found": bool(q.get("expect_not_found")),
            "not_found": not_found,
            "review": verdict,
            "latency_s": round(latency, 1),
            "tokens": resp.token_usage["total_tokens"],
            "tool_calls": [t["tool"] for t in resp.tool_calls],
        })
        tag = "oob" if q.get("expect_not_found") else "doc"
        print(f"  {'✓' if hit else '✗'} [{tag}] {q['question'][:30]} | {latency:.0f}s | {[t['tool'] for t in resp.tool_calls]}")

    n = len(questions)
    return {
        "multi_agent": multi_agent,
        "num_questions": n,
        "source_hit_rate": hits / n if n else 0.0,
        "hallucination_free_rate": honest / n if n else 0.0,
        "avg_latency_s": round(total_latency / n, 1) if n else 0.0,
        "avg_tokens": round(total_tokens / n) if n else 0,
        "review_verdicts": verdicts if multi_agent else None,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent 级评测：单 Agent vs 多 Agent 对比")
    parser.add_argument("--questions", type=Path, default=EVAL_QUESTIONS_PATH)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    parser.add_argument("--group", choices=["single", "multi", "both"], default="both",
                        help="跑哪一组（调 prompt 后只复测受影响的一组时用）")
    parser.add_argument("--output", type=Path, default=Path("data/eval_agent_report.json"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit(f"{args.questions} 为空或不存在")

    run_multi = [False, True] if args.group == "both" else [args.group == "multi"]
    print(f"{len(questions)} 条评测问题，Agent 已加载，开始评测（{len(run_multi)} 组）...")

    groups = []
    for ma in run_multi:
        print(f"\n===== {'多 Agent（规划员+审校员）' if ma else '单 Agent'} 组 =====")
        groups.append(run_group(questions, multi_agent=ma))
        g = groups[-1]
        print(f"来源命中率: {g['source_hit_rate']:.2%} | 平均延迟: {g['avg_latency_s']}s | 平均 token: {g['avg_tokens']}")
        if g["review_verdicts"]:
            print(f"审校结论: {g['review_verdicts']}")

    report = {"groups": groups}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存 → {args.output}")


if __name__ == "__main__":
    main()
