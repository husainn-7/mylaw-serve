import os

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class TorchEmbedder:
    """bge-small embeddings with PyTorch (the MyLaw-RAG baseline)."""

    def __init__(self, model_name=EMBED_MODEL, device="cpu"):
        from sentence_transformers import SentenceTransformer

        self.name = f"torch-fp32-{device}"
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts, batch_size=32):
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


class TorchReranker:
    """MiniLM cross-encoder with PyTorch (the MyLaw-RAG baseline)."""

    def __init__(self, model_name=RERANK_MODEL, device="cpu"):
        from sentence_transformers import CrossEncoder

        self.name = f"torch-fp32-{device}"
        self.model = CrossEncoder(model_name, max_length=512, device=device)

    def score(self, query, docs, batch_size=16):
        if not docs:
            return []
        scores = self.model.predict([(query, d) for d in docs], batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def load_engines(engine=None, device=None):
    engine = engine or os.environ.get("ENGINE", "torch")
    device = device or os.environ.get("DEVICE", "cpu")

    threads = os.environ.get("TORCH_THREADS")
    if threads and engine == "torch":
        import torch

        torch.set_num_threads(int(threads))

    if engine == "torch":
        return TorchEmbedder(device=device), TorchReranker(device=device)
    if engine == "fake":
        from mlserve.fake import FakeEmbedder, FakeReranker

        return FakeEmbedder(), FakeReranker()
    raise ValueError(f"unknown engine {engine!r}. Available: torch, fake")
