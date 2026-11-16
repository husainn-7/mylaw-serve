"""Load test a running server with N concurrent clients.

    uvicorn mlserve.api:app --port 8000        (in one terminal)
    python bench/loadtest.py --concurrency 1,4,8 --duration 15
"""
import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlserve.envinfo import env_info  # noqa: E402
from mlserve.passages import load_passages  # noqa: E402
from mlserve.stats import summarize  # noqa: E402


def make_payloads(endpoint, questions, docs):
    if endpoint == "search":
        return [{"query": q} for q in questions]
    if endpoint == "embed":
        return [{"texts": [q]} for q in questions]
    if endpoint == "rerank":
        return [{"query": q, "docs": docs} for q in questions]
    raise ValueError(f"unknown endpoint {endpoint!r}")


def run_level(base_url, path, payloads, concurrency, duration, warmup):
    start = time.perf_counter()
    record_from = start + warmup
    stop_at = record_from + duration
    latencies, errors, lock = [], [0], threading.Lock()

    def worker(seed):
        rng = random.Random(seed)
        client = httpx.Client(base_url=base_url, timeout=120)
        while time.perf_counter() < stop_at:
            payload = payloads[rng.randrange(len(payloads))]
            t0 = time.perf_counter()
            try:
                ok = client.post(path, json=payload).status_code == 200
            except Exception:
                ok = False
            ms = (time.perf_counter() - t0) * 1000
            if t0 >= record_from:
                with lock:
                    if ok:
                        latencies.append(ms)
                    else:
                        errors[0] += 1
        client.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    return {
        "concurrency": concurrency,
        "errors": errors[0],
        "rps": round(len(latencies) / duration, 2),
        **summarize(latencies),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoint", default="search", choices=["search", "embed", "rerank"])
    ap.add_argument("--concurrency", default="1,4,8", help="comma separated, e.g. 1,4,8")
    ap.add_argument("--duration", type=float, default=15, help="seconds measured per level")
    ap.add_argument("--warmup", type=float, default=3, help="seconds not measured per level")
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args(argv)

    try:
        health = httpx.get(f"{args.url}/healthz", timeout=10).json()
    except Exception as e:
        raise SystemExit(f"cannot reach {args.url}/healthz ({e}). Is the server running?")

    questions = [q["question"] for q in map(json.loads, open(args.questions, encoding="utf-8")) if q["answerable"]]
    passages = load_passages(args.chunks)
    docs = [p["text"] for p in passages[:: max(1, len(passages) // 30)]][:30]
    payloads = make_payloads(args.endpoint, questions, docs)

    label = args.label or f"{health['embedder']}_{args.endpoint}"
    levels = []
    print(f"\n=== load test: {label} -> /{args.endpoint} ({args.duration:.0f}s per level) ===")
    print(f"{'clients':>7s} {'requests':>9s} {'errors':>7s} {'req/s':>8s} {'p50 ms':>9s} {'p95 ms':>9s} {'p99 ms':>9s}")
    for c in [int(x) for x in args.concurrency.split(",")]:
        r = run_level(args.url, f"/{args.endpoint}", payloads, c, args.duration, args.warmup)
        levels.append(r)
        print(f"{c:7d} {r['n']:9d} {r['errors']:7d} {r['rps']:8.2f} {r['p50'] or 0:9.1f} {r['p95'] or 0:9.1f} {r['p99'] or 0:9.1f}", flush=True)

    result = {
        "label": label,
        "endpoint": args.endpoint,
        "server": health,
        "duration_s": args.duration,
        "levels": levels,
        "client_env": env_info(),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"load_{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_dir / f'load_{label}.json'}")
    return result


if __name__ == "__main__":
    main()
