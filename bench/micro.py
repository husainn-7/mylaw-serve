"""Model-level latency without HTTP: how fast is each model on its own, at different batch sizes?

    python bench/micro.py --engine torch --runs 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlserve.engines import QUERY_PREFIX, load_engines  # noqa: E402
from mlserve.envinfo import env_info  # noqa: E402
from mlserve.passages import doc_text, load_passages  # noqa: E402
from mlserve.stats import summarize  # noqa: E402


def rss_mb():
    return round(psutil.Process().memory_info().rss / 1e6, 1)


def timed(fn, runs, warmup):
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t) * 1000)
    return latencies


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args(argv)

    rss_before = rss_mb()
    t = time.perf_counter()
    embedder, reranker = load_engines(args.engine, args.device)
    load_s = round(time.perf_counter() - t, 2)
    rss_loaded = rss_mb()

    questions = [q["question"] for q in map(json.loads, open(args.questions, encoding="utf-8")) if q["answerable"]]
    passages = load_passages(args.chunks)
    step = max(1, len(passages) // 32)
    docs = [doc_text(p) for p in passages[::step]][:32]
    rows = []

    for bs in (1, 8, 32):
        batch = [QUERY_PREFIX + questions[i % len(questions)] for i in range(bs)]
        lat = timed(lambda: embedder.embed(batch), args.runs, args.warmup)
        rows.append({"task": f"embed {bs} queries", "batch": bs, **summarize(lat)})

    doc_batch = (docs * 2)[:32]
    lat = timed(lambda: embedder.embed(doc_batch), max(5, args.runs // 3), 1)
    rows.append({"task": "embed 32 passages", "batch": 32, **summarize(lat)})

    for n in (10, 30):
        subset = (docs * 2)[:n]
        lat = timed(lambda: reranker.score(questions[0], subset), args.runs, args.warmup)
        rows.append({"task": f"rerank {n} passages", "batch": n, **summarize(lat)})

    label = args.label or embedder.name
    result = {
        "label": label,
        "embedder": embedder.name,
        "reranker": reranker.name,
        "model_load_s": load_s,
        "rss_mb_before": rss_before,
        "rss_mb_after_load": rss_loaded,
        "rss_mb_after_runs": rss_mb(),
        "runs": args.runs,
        "env": env_info(),
        "results": rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"micro_{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== micro benchmark: {label} ===")
    print(f"model load {load_s}s, memory {rss_loaded} MB after load, {result['rss_mb_after_runs']} MB after runs")
    print(f"{'task':22s} {'p50 ms':>9s} {'p95 ms':>9s} {'ms/item':>9s}")
    for r in rows:
        print(f"{r['task']:22s} {r['p50']:9.2f} {r['p95']:9.2f} {r['p50'] / r['batch']:9.2f}")
    print(f"\nsaved -> {out_dir / f'micro_{label}.json'}")
    return result


if __name__ == "__main__":
    main()
