import json
import socket
import threading
import time

import pytest
import uvicorn

from bench import loadtest, micro, quality
from mlserve.api import create_app
from mlserve.fake import FakeEmbedder, FakeReranker
from mlserve.passages import load_passages
from mlserve.service import SearchService


def test_quality_script_runs_and_saves_results(corpus, tmp_path):
    out = tmp_path / "results"
    result = quality.main([
        "--engine", "fake", "--chunks", corpus["chunks"], "--questions", corpus["questions"],
        "--index-dir", str(tmp_path / "index"), "--out", str(out),
    ])
    assert result["metrics"]["all"]["n"] == 6  # the out-of-scope question is skipped
    assert result["metrics"]["all"]["hit@5"] >= 0.8
    saved = json.loads((out / "quality_fake.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["easy"]["n"] == 3 and saved["metrics"]["hard"]["n"] == 3
    assert saved["sequential_latency_ms"]["n"] == 6


def test_micro_script_runs_and_saves_results(corpus, tmp_path):
    out = tmp_path / "results"
    result = micro.main([
        "--engine", "fake", "--chunks", corpus["chunks"], "--questions", corpus["questions"],
        "--runs", "3", "--warmup", "1", "--out", str(out),
    ])
    tasks = [r["task"] for r in result["results"]]
    assert "embed 1 queries" in tasks and "rerank 30 passages" in tasks
    assert all(r["p50"] is not None for r in result["results"])
    assert (out / "micro_fake.json").exists()


@pytest.fixture
def server(corpus, tmp_path):
    svc = SearchService.build(FakeEmbedder(), FakeReranker(), load_passages(corpus["chunks"]), index_dir=tmp_path / "index", pool=10)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(svc), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.mark.parametrize("endpoint", ["search", "embed", "rerank"])
def test_loadtest_against_a_live_server(server, corpus, tmp_path, endpoint):
    out = tmp_path / "results"
    result = loadtest.main([
        "--url", server, "--endpoint", endpoint, "--concurrency", "1,3", "--duration", "1", "--warmup", "0",
        "--questions", corpus["questions"], "--chunks", corpus["chunks"], "--out", str(out),
    ])
    assert [lv["concurrency"] for lv in result["levels"]] == [1, 3]
    for lv in result["levels"]:
        assert lv["errors"] == 0 and lv["rps"] > 0 and lv["p50"] is not None
    assert result["server"]["embedder"] == "fake"


def test_loadtest_explains_when_the_server_is_down(corpus, tmp_path):
    with pytest.raises(SystemExit) as e:
        loadtest.main(["--url", "http://127.0.0.1:9", "--questions", corpus["questions"], "--chunks", corpus["chunks"], "--out", str(tmp_path)])
    assert "Is the server running" in str(e.value)


def test_quality_pool_option_changes_the_label_and_is_recorded(corpus, tmp_path):
    out = tmp_path / "results"
    result = quality.main([
        "--engine", "fake", "--chunks", corpus["chunks"], "--questions", corpus["questions"],
        "--index-dir", str(tmp_path / "index"), "--out", str(out), "--pool", "3",
    ])
    assert result["label"] == "fake_pool3" and result["pool"] == 3
    assert (out / "quality_fake_pool3.json").exists()
    assert result["metrics"]["all"]["n"] == 6
