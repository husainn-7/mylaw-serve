"""Tiny word-hashing engines. Used by the tests and by CI so no model downloads are needed.
They are NOT meant to give good retrieval quality."""
import re
import zlib

import numpy as np

DIM = 256


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEmbedder:
    name = "fake"

    def embed(self, texts, batch_size=32):
        out = np.zeros((len(texts), DIM), dtype="float32")
        for i, t in enumerate(texts):
            for w in _words(t):
                out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class FakeReranker:
    name = "fake"

    def score(self, query, docs):
        q = set(_words(query))
        return [float(len(q & set(_words(d)))) for d in docs]
