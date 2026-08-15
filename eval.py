import argparse
import json
from pathlib import Path

from config import RETRIEVAL_SCORE_THRESHOLD
from llm_providers import rewrite_query
from vector_stores import get_vector_store

EVAL_QUESTIONS_PATH = Path("data/eval_questions.jsonl")


def load_questions(path: Path) -> list[dict]:
    questions = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def evaluate(store, questions: list[dict], k: int, threshold: float, rewrite: bool = False) -> dict:
    details = []
    hits = 0
    rr_sum = 0.0

    for q in questions:
        expected = set(q["expected_sources"])
        query_text = rewrite_query(q["question"]) if rewrite else q["question"]
        results = store.similarity_search(query_text, k=k, score_threshold=threshold)
        sources = [r["source"] for r in results]

        rank = next((i + 1 for i, s in enumerate(sources) if s in expected), None)
        hit = rank is not None
        if hit:
            hits += 1
            rr_sum += 1.0 / rank

        details.append({
            "question": q["question"],
            "query_text": query_text,
            "expected_sources": q["expected_sources"],
            "retrieved_sources": sources,
            "scores": [r["similarity"] for r in results],
            "hit": hit,
            "rank": rank,
        })

    n = len(questions)
    return {
        "k": k,
        "threshold": threshold,
        "rewrite": rewrite,
        "num_questions": n,
        "recall_at_k": hits / n if n else 0.0,
        "mrr_at_k": rr_sum / n if n else 0.0,
        "details": details,
    }


def print_report(report: dict, verbose: bool) -> None:
    print(f"\n===== 检索评测结果 =====")
    print(f"问题数: {report['num_questions']} | k={report['k']} | 阈值={report['threshold']} | 查询改写={'开' if report['rewrite'] else '关'}")
    print(f"Recall@{report['k']}: {report['recall_at_k']:.2%}")
    print(f"MRR@{report['k']}:    {report['mrr_at_k']:.4f}")

    misses = [d for d in report["details"] if not d["hit"]]
    print(f"\n未命中 ({len(misses)}):")
    for d in misses:
        print(f"  ✗ {d['question']}")
        print(f"    期望: {d['expected_sources']}")
        print(f"    实际: {d['retrieved_sources'][:3]} scores={d['scores'][:3]}")

    if verbose:
        print(f"\n全部明细:")
        for d in report["details"]:
            mark = "✓" if d["hit"] else "✗"
            print(f"  {mark} rank={d['rank']} {d['question']} -> {d['retrieved_sources'][:2]}")


def main():
    parser = argparse.ArgumentParser(description="检索质量评测：Recall@k / MRR@k")
    parser.add_argument("--questions", type=Path, default=EVAL_QUESTIONS_PATH)
    parser.add_argument("--k", type=int, default=5, help="top-k（默认 5）")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help=f"相关度阈值（默认 0=不过滤，用于测原始检索能力；线上生效值 {RETRIEVAL_SCORE_THRESHOLD}）")
    parser.add_argument("--output", type=Path, help="把完整报告存为 JSON")
    parser.add_argument("--rewrite", action="store_true", help="检索前用 LLM 改写查询（每题一次 LLM 调用）")
    parser.add_argument("--verbose", action="store_true", help="打印每题明细")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit(f"{args.questions} 为空或不存在")
    print(f"{len(questions)} 条评测问题，加载向量库...")

    store = get_vector_store()
    stats = store.get_stats()
    print(f"向量存储: {stats['backend']}, 存量: {stats.get('total_chunks', 0)} 块")

    report = evaluate(store, questions, k=args.k, threshold=args.threshold, rewrite=args.rewrite)
    print_report(report, verbose=args.verbose)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已保存 → {args.output}")


if __name__ == "__main__":
    main()
