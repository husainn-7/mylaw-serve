import json

from bench import report


def write(d, name, data):
    (d / name).write_text(json.dumps(data), encoding="utf-8")


def make_variant(d, label, hard_mrr, rerank_p50, rps1, rpsmax):
    write(d, f"quality_{label}.json", {
        "metrics": {"hard": {"hit@1": 0.3, "hit@5": 0.6, "mrr@10": hard_mrr}, "easy": {"mrr@10": 0.9}},
        "sequential_latency_ms": {"p50": rerank_p50 + 10},
    })
    write(d, f"micro_{label}.json", {"results": [{"task": "rerank 30 passages", "p50": rerank_p50}], "rss_mb_after_runs": 1200.0})
    write(d, f"load_{label}_search.json", {"levels": [{"concurrency": 4, "rps": rpsmax}, {"concurrency": 1, "rps": rps1}]})


def test_report_puts_torch_first_and_computes_deltas(tmp_path):
    make_variant(tmp_path, "onnx-int8-cpu", 0.48, 300.0, 3.0, 3.5)
    make_variant(tmp_path, "torch-fp32-cpu", 0.51, 900.0, 1.0, 1.1)
    rows = report.collect(tmp_path)
    assert [r["variant"] for r in rows] == ["torch-fp32-cpu", "onnx-int8-cpu"]
    assert rows[0]["d_hard_mrr"] == 0.0
    assert rows[1]["d_hard_mrr"] == -0.03
    assert rows[1]["rps_1"] == 3.0 and rows[1]["rps_max"] == 3.5  # picks lowest and highest concurrency


def test_report_survives_missing_files_and_formats_both_ways(tmp_path):
    make_variant(tmp_path, "torch-fp32-cpu", 0.51, 900.0, 1.0, 1.1)
    write(tmp_path, "quality_onnx-fp32-cpu.json", {
        "metrics": {"hard": {"hit@1": 0.3, "hit@5": 0.6, "mrr@10": 0.5}, "easy": {"mrr@10": 0.9}},
        "sequential_latency_ms": {"p50": 400.0},
    })
    rows = report.collect(tmp_path)
    plain = report.format_table(rows)
    md = report.format_table(rows, markdown=True)
    assert "onnx-fp32-cpu" in plain and "-" in plain
    assert md.startswith("| variant |") and md.count("\n") == 3
