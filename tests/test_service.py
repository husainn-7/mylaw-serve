import mlserve.service as service_module
from mlserve.fake import FakeEmbedder, FakeReranker
from mlserve.passages import load_passages
from mlserve.service import SearchService


def make(corpus, index_dir, **kw):
    return SearchService.build(FakeEmbedder(), FakeReranker(), load_passages(corpus["chunks"]), index_dir=index_dir, pool=10, **kw)


def test_search_finds_the_right_section_first(corpus, tmp_path):
    svc = make(corpus, tmp_path / "index")
    sections, timings = svc.search("How many days of annual leave does an employee get?", top_sections=3)
    assert sections[0]["id"] == "employment_act-60E"
    assert sections[0]["label"] == "Employment Act 1955 s.60E"
    assert len(sections) <= 3
    assert set(timings) == {"embed_ms", "faiss_ms", "rerank_ms", "total_ms"}


def test_sections_are_deduplicated(corpus, tmp_path):
    svc = make(corpus, tmp_path / "index")
    ranked, _ = svc.rank_passages("term5 means a thing defined in this Act")
    sections, _ = svc.search("term5 means a thing defined in this Act", top_sections=10)
    ids = [s["id"] for s in sections]
    assert len(ids) == len(set(ids))
    assert len(ranked) >= len(ids)


def test_index_is_cached_per_engine_and_reused(corpus, tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    make(corpus, index_dir)
    assert (index_dir / "fake.faiss").exists()

    def boom(*a, **k):
        raise AssertionError("index should have been loaded from disk")

    monkeypatch.setattr(service_module, "build_index", boom)
    make(corpus, index_dir)  # must not rebuild


def test_index_is_rebuilt_when_the_corpus_changes(corpus, tmp_path):
    index_dir = tmp_path / "index"
    svc = make(corpus, index_dir)
    passages = load_passages(corpus["chunks"])[:-2]
    svc2 = SearchService.build(FakeEmbedder(), FakeReranker(), passages, index_dir=index_dir, pool=10)
    assert svc2.index.ntotal == len(passages) != svc.index.ntotal
