import pytest
from fastapi.testclient import TestClient

from mlserve.api import create_app
from mlserve.fake import FakeEmbedder, FakeReranker
from mlserve.passages import load_passages
from mlserve.service import SearchService


@pytest.fixture
def client(corpus, tmp_path):
    svc = SearchService.build(FakeEmbedder(), FakeReranker(), load_passages(corpus["chunks"]), index_dir=tmp_path / "index", pool=10)
    with TestClient(create_app(svc)) as c:
        yield c


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["embedder"] == "fake" and body["passages"] > 0


def test_search(client):
    r = client.post("/search", json={"query": "How many days of annual leave does an employee get?", "top_sections": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["sections"][0]["id"] == "employment_act-60E"
    assert len(body["sections"]) == 2
    assert body["timings"]["total_ms"] >= 0


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "x", "top_sections": 11}, {"top_sections": 3}, {"query": "x" * 501}])
def test_search_validation(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_embed(client):
    r = client.post("/embed", json={"texts": ["a b c", "d e f"]}).json()
    assert r["count"] == 2 and r["dim"] > 0 and "vectors" not in r
    r = client.post("/embed", json={"texts": ["a b c"], "return_vectors": True}).json()
    assert len(r["vectors"]) == 1 and len(r["vectors"][0]) == r["dim"]


def test_embed_validation(client):
    assert client.post("/embed", json={"texts": []}).status_code == 422
    assert client.post("/embed", json={"texts": ["x"] * 65}).status_code == 422


def test_rerank(client):
    r = client.post("/rerank", json={"query": "annual leave days", "docs": ["annual leave days", "something unrelated"]}).json()
    assert r["scores"][0] > r["scores"][1]
