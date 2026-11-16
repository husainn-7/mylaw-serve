import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlserve.engines import load_engines
from mlserve.passages import load_passages
from mlserve.service import SearchService


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_sections: int = Field(3, ge=1, le=10)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    return_vectors: bool = False


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    docs: list[str] = Field(min_length=1, max_length=64)


def load_default_service():
    embedder, reranker = load_engines()
    passages = load_passages(os.environ.get("CHUNKS", "data/chunks.jsonl"))
    return SearchService.build(embedder, reranker, passages, index_dir=os.environ.get("INDEX_DIR", "index"))


def create_app(service=None):
    state = {"service": service}

    @asynccontextmanager
    async def lifespan(app):
        if state["service"] is None:
            state["service"] = load_default_service()
        yield

    app = FastAPI(title="MyLaw-Serve", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        svc = state["service"]
        return {
            "status": "ok",
            "embedder": svc.embedder.name,
            "reranker": svc.reranker.name,
            "passages": len(svc.passages),
            "pool": svc.pool,
        }

    @app.post("/search")
    def search(req: SearchRequest):
        sections, timings = state["service"].search(req.query, req.top_sections)
        return {"query": req.query, "sections": sections, "timings": timings}

    @app.post("/embed")
    def embed(req: EmbedRequest):
        t = time.perf_counter()
        vecs = state["service"].embedder.embed(req.texts)
        out = {"count": len(req.texts), "dim": int(vecs.shape[1]), "latency_ms": round((time.perf_counter() - t) * 1000, 2)}
        if req.return_vectors:
            out["vectors"] = vecs.tolist()
        return out

    @app.post("/rerank")
    def rerank(req: RerankRequest):
        t = time.perf_counter()
        scores = state["service"].reranker.score(req.query, req.docs)
        return {"scores": scores, "latency_ms": round((time.perf_counter() - t) * 1000, 2)}

    return app


app = create_app()
