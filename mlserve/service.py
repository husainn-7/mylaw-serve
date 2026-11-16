import time
from pathlib import Path

from mlserve.engines import QUERY_PREFIX
from mlserve.metrics import section_id
from mlserve.passages import doc_text


def build_index(embedder, passages, batch_size=32):
    import faiss

    emb = embedder.embed([doc_text(p) for p in passages], batch_size=batch_size)
    index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine on normalized vectors
    index.add(emb)
    return index


class SearchService:
    """Dense search over passages, then cross-encoder reranking. Same pipeline as MyLaw-RAG."""

    def __init__(self, embedder, reranker, passages, index, pool=30):
        self.embedder = embedder
        self.reranker = reranker
        self.passages = passages
        self.index = index
        self.pool = pool

    @classmethod
    def build(cls, embedder, reranker, passages, index_dir="index", pool=30, rebuild=False):
        """Each embedding model needs its own index, so the file is named after the embedder."""
        import faiss

        path = Path(index_dir) / f"{embedder.name}.faiss"
        index = None
        if path.exists() and not rebuild:
            index = faiss.read_index(str(path))
            if index.ntotal != len(passages):
                index = None
        if index is None:
            index = build_index(embedder, passages)
            path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(path))
        return cls(embedder, reranker, passages, index, pool)

    def rank_passages(self, query):
        """Return all candidate passages sorted by rerank score, plus a per-stage timing breakdown."""
        t0 = time.perf_counter()
        qvec = self.embedder.embed([QUERY_PREFIX + query])
        t1 = time.perf_counter()
        _, ids = self.index.search(qvec, self.pool)
        t2 = time.perf_counter()
        cands = [self.passages[i] for i in ids[0] if i >= 0]
        scores = self.reranker.score(query, [doc_text(c) for c in cands])
        t3 = time.perf_counter()
        ranked = sorted(zip(scores, cands), key=lambda x: -x[0])
        timings = {
            "embed_ms": round((t1 - t0) * 1000, 2),
            "faiss_ms": round((t2 - t1) * 1000, 2),
            "rerank_ms": round((t3 - t2) * 1000, 2),
            "total_ms": round((t3 - t0) * 1000, 2),
        }
        return [{**c, "score": float(s)} for s, c in ranked], timings

    def search(self, query, top_sections=3):
        ranked, timings = self.rank_passages(query)
        sections = {}
        for p in ranked:
            sid = section_id(p["id"])
            entry = sections.setdefault(sid, {
                "id": sid,
                "label": f"{p['act']} s.{p['section']}",
                "heading": p["heading"],
                "score": p["score"],
                "passages": [],
            })
            entry["passages"].append(p["text"])
        return list(sections.values())[:top_sections], timings
