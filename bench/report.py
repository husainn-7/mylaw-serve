"""Put every variant's quality, latency and throughput side by side.

    python bench/report.py                      # every variant found in bench/results
    python bench/report.py torch-fp32-cpu onnx-fp32-cpu
    python bench/report.py --markdown           # table ready to paste into the README
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLUMNS = [
    ("variant", "variant"),
    ("hard_hit1", "hard Hit@1"),
    ("hard_hit5", "hard Hit@5"),
    ("hard_mrr", "hard MRR"),
    ("easy_mrr", "easy MRR"),
    ("d_hard_mrr", "d hard MRR"),
    ("query_p50", "query p50 ms"),
    ("rerank_p50", "rerank-30 p50 ms"),
    ("rps_1", "req/s (1 client)"),
    ("rps_max", "req/s (max clients)"),
    ("ram", "RAM MB"),
]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def collect(results_dir="bench/results", labels=None):
    d = Path(results_dir)
    if not labels:
        labels = sorted(p.stem[len("quality_"):] for p in d.glob("quality_*.json"))
        labels.sort(key=lambda x: (not x.startswith("torch"), x))  # torch baseline first
    rows = []
    for label in labels:
        q = _read(d / f"quality_{label}.json")
        m = _read(d / f"micro_{label}.json")
        loads = sorted(d.glob(f"load_{label}_search.json"))
        load = _read(loads[0]) if loads else None
        row = {"variant": label}
        if q:
            row["hard_hit1"] = q["metrics"]["hard"]["hit@1"]
            row["hard_hit5"] = q["metrics"]["hard"]["hit@5"]
            row["hard_mrr"] = q["metrics"]["hard"]["mrr@10"]
            row["easy_mrr"] = q["metrics"]["easy"]["mrr@10"]
            row["query_p50"] = q["sequential_latency_ms"]["p50"]
        if m:
            rerank = [r for r in m["results"] if r["task"] == "rerank 30 passages"]
            row["rerank_p50"] = rerank[0]["p50"] if rerank else None
            row["ram"] = m["rss_mb_after_runs"]
        if load:
            levels = sorted(load["levels"], key=lambda x: x["concurrency"])
            row["rps_1"] = levels[0]["rps"]
            row["rps_max"] = levels[-1]["rps"]
        rows.append(row)
    base = rows[0].get("hard_mrr") if rows else None
    for r in rows:
        if base is not None and r.get("hard_mrr") is not None:
            r["d_hard_mrr"] = round(r["hard_mrr"] - base, 3)
    return rows


def _fmt(key, value):
    if value is None:
        return "-"
    if key == "d_hard_mrr":
        return f"{value:+.3f}"
    if key in ("hard_hit1", "hard_hit5", "hard_mrr", "easy_mrr"):
        return f"{value:.3f}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def format_table(rows, markdown=False):
    head = [title for _, title in COLUMNS]
    body = [[_fmt(k, r.get(k)) for k, _ in COLUMNS] for r in rows]
    if markdown:
        lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines)
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(head)] if body else [len(h) for h in head]
    fmt = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    return "\n".join([fmt(head), fmt(["-" * w for w in widths])] + [fmt(row) for row in body])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--dir", default="bench/results")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)
    rows = collect(args.dir, args.labels or None)
    if not rows:
        raise SystemExit(f"no quality_*.json files in {args.dir}")
    table = format_table(rows, args.markdown)
    print(table)
    return table


if __name__ == "__main__":
    main()
