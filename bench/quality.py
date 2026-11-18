"""Does the serving pipeline still retrieve as well as MyLaw-RAG? Run this for every engine variant.

    python bench/quality.py --engine torch
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlserve.engines import combine_names, load_engines  # noqa: E402
from mlserve.envinfo import env_info  # noqa: E402
from mlserve.metrics import collapse, first_gold_rank, metrics_for  # noqa: E402
from mlserve.passages import load_passages  # noqa: E402
from mlserve.service import SearchService  # noqa: E402
from mlserve.stats import summarize  # noqa: E402

# MyLaw-RAG results for dense + MiniLM rerank (the pipeline this service re-implements)
PROJECT1 = {
    "easy": {"hit@1": 0.843, "hit@3": 0.961, "hit@5": 0.980, "hit@10": 0.980, "mrr@10": 0.904},
    "hard": {"hit@1": 0.357, "hit@3": 0.643, "hit@5": 0.679, "hit@10": 0.821, "mrr@10": 0.510},
}
COLS = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"]


def evaluate(service, questions):
    rows = []
    totals = []
    for q in questions:
        ranked, timings = service.rank_passages(q["question"])
        sections = collapse([p["id"] for p in ranked])
        rows.append({
            "id": q["id"],
            "split": q.get("split", "easy"),
            "rank": first_gold_rank(sections, set(q["gold"])),
        })
        totals.append(timings["total_ms"])
    out = {
        "all": metrics_for([r["rank"] for r in rows]),
        "easy": metrics_for([r["rank"] for r in rows if r["split"] == "easy"]),
        "hard": metrics_for([r["rank"] for r in rows if r["split"] == "hard"]),
    }
    return out, rows, summarize(totals)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--pool", type=int, default=30, help="how many dense candidates the reranker scores")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args(argv)

    embedder, reranker = load_engines(args.engine, args.device)
    passages = load_passages(args.chunks)
    service = SearchService.build(embedder, reranker, passages, index_dir=args.index_dir, rebuild=args.rebuild, pool=args.pool)

    questions = [q for q in map(json.loads, open(args.questions, encoding="utf-8")) if q["answerable"]]
    metrics, rows, latency = evaluate(service, questions)

    label = args.label or combine_names(embedder.name, reranker.name)
    if args.pool != 30 and not args.label:
        label += f"_pool{args.pool}"
    result = {
        "label": label,
        "embedder": embedder.name,
        "reranker": reranker.name,
        "pool": args.pool,
        "metrics": metrics,
        "sequential_latency_ms": latency,
        "env": env_info(),
        "rows": rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"quality_{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== quality: {label} ({len(questions)} answerable questions) ===")
    print(f"{'split':6s} {'n':>3s}" + "".join(f"{c:>9s}" for c in COLS))
    for split in ("all", "easy", "hard"):
        m = metrics[split]
        if m:
            print(f"{split:6s} {m['n']:3d}" + "".join(f"{m[c]:9.3f}" for c in COLS))

    if metrics["easy"] and metrics["hard"]:
        print("\ndifference from the MyLaw-RAG baseline (dense + MiniLM rerank), same questions:")
        for split in ("easy", "hard"):
            diffs = "".join(f"{metrics[split][c] - PROJECT1[split][c]:+9.3f}" for c in COLS)
            print(f"{split:6s}    " + diffs)
        print("(0.000 means identical. One easy question is worth 0.020, one hard question 0.036.)")
    print(f"\nsequential latency per query: p50 {latency['p50']} ms, p95 {latency['p95']} ms")
    print(f"saved -> {out_dir / f'quality_{label}.json'}")
    return result


if __name__ == "__main__":
    main()
