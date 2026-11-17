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


def combine_names(embedder_name, reranker_name):
    """One label for a pair of engines, used in result file names and tables."""
    return embedder_name if embedder_name == reranker_name else f"{embedder_name}_{reranker_name}"


ENGINES = ["torch", "onnx", "onnx-int8-rerank", "onnx-int8", "fake"]


def load_engines(engine=None, device=None):
    engine = engine or os.environ.get("ENGINE", "torch")
    device = device or os.environ.get("DEVICE", "cpu")

    if engine == "torch":
        threads = os.environ.get("TORCH_THREADS")
        if threads:
            import torch

            torch.set_num_threads(int(threads))
        return TorchEmbedder(device=device), TorchReranker(device=device)

    if engine in ("onnx", "onnx-int8", "onnx-int8-rerank"):
        from mlserve.onnx_engines import OnnxEmbedder, OnnxReranker

        root = os.environ.get("MODELS_DIR", "models/onnx")
        embed_variant = "int8" if engine == "onnx-int8" else "fp32"
        rerank_variant = "fp32" if engine == "onnx" else "int8"
        return OnnxEmbedder(f"{root}/embedder", embed_variant), OnnxReranker(f"{root}/reranker", rerank_variant)

    if engine == "fake":
        from mlserve.fake import FakeEmbedder, FakeReranker

        return FakeEmbedder(), FakeReranker()

    raise ValueError(f"unknown engine {engine!r}. Available: {', '.join(ENGINES)}")
