import numpy as np


def summarize(latencies_ms):
    """Latency summary in milliseconds."""
    if len(latencies_ms) == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    a = np.asarray(latencies_ms, dtype="float64")
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "min": round(float(a.min()), 2),
        "max": round(float(a.max()), 2),
    }
