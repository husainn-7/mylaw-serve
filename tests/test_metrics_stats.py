from mlserve.metrics import collapse, first_gold_rank, metrics_for, section_id
from mlserve.stats import summarize


def test_section_id_strips_passage_suffix():
    assert section_id("employment_act-60E-p3") == "employment_act-60E"


def test_collapse_keeps_first_passage_per_section():
    ids = ["x-1-p0", "x-1-p1", "x-2-p0", "x-1-p2", "x-3-p0"]
    assert collapse(ids) == ["x-1", "x-2", "x-3"]
    assert collapse(ids, k=2) == ["x-1", "x-2"]


def test_first_gold_rank():
    assert first_gold_rank(["a", "b", "c"], {"c"}) == 3
    assert first_gold_rank(["a"], {"z"}) is None


def test_metrics_for():
    m = metrics_for([1, 3, None, 6])
    assert (m["hit@1"], m["hit@3"], m["hit@5"], m["hit@10"]) == (0.25, 0.5, 0.5, 0.75)
    assert m["mrr@10"] == 0.375
    assert metrics_for([]) is None


def test_summarize_percentiles():
    s = summarize(list(range(1, 101)))
    assert s["n"] == 100 and s["p50"] == 50.5 and s["p95"] == 95.05 and s["min"] == 1 and s["max"] == 100
    assert summarize([])["n"] == 0
