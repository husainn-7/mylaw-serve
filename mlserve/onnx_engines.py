import os
from pathlib import Path

import numpy as np


def _model_file(model_dir, variant):
    f = Path(model_dir) / ("model_int8.onnx" if variant == "int8" else "model.onnx")
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Create the ONNX models first:  python -m mlserve.onnx_export")
    return f


def _session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    threads = os.environ.get("ORT_THREADS")
    if threads:
        options.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _feeds(session, enc):
    wanted = {i.name for i in session.get_inputs()}
    return {k: np.asarray(enc[k], dtype="int64") for k in wanted if k in enc}


class OnnxEmbedder:
    """bge-small with ONNX Runtime on CPU: CLS token, then L2 normalisation (what sentence-transformers does)."""

    def __init__(self, model_dir, variant="fp32"):
        path = _model_file(model_dir, variant)
        from transformers import AutoTokenizer

        self.name = f"onnx-{variant}-cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = _session(path)

    def embed(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="np")
            hidden = self.session.run(None, _feeds(self.session, enc))[0]
            cls = hidden[:, 0].astype("float32")
            out.append(cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12))
        return np.vstack(out)


class OnnxReranker:
    """MiniLM cross-encoder with ONNX Runtime on CPU. Scores go through a sigmoid like the PyTorch version."""

    def __init__(self, model_dir, variant="fp32"):
        path = _model_file(model_dir, variant)
        from transformers import AutoTokenizer

        self.name = f"onnx-{variant}-cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = _session(path)

    def score(self, query, docs, batch_size=16):
        scores = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            enc = self.tokenizer([query] * len(batch), batch, padding=True, truncation=True, max_length=512, return_tensors="np")
            logits = self.session.run(None, _feeds(self.session, enc))[0].reshape(-1).astype("float64")
            scores.extend((1.0 / (1.0 + np.exp(-logits))).tolist())
        return scores
