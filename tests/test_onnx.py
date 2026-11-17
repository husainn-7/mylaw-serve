"""ONNX parity tests on tiny random models. Skipped automatically when the ONNX libraries are not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")

from mlserve.onnx_engines import OnnxEmbedder, OnnxReranker  # noqa: E402
from mlserve.onnx_export import export_one  # noqa: E402

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + (
    "the a of to and in is for on employee leave annual days data consent contract coercion act section shall be "
    "entitled paid eight twelve months service personal without".split()
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    (root / "vocab.txt").write_text("\n".join(WORDS), encoding="utf-8")
    tok = transformers.BertTokenizerFast(str(root / "vocab.txt"), do_lower_case=True)
    cfg = dict(vocab_size=len(WORDS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
               intermediate_size=64, max_position_embeddings=64)
    torch.manual_seed(0)
    emb = transformers.BertModel(transformers.BertConfig(**cfg), add_pooling_layer=False).eval()
    rr = transformers.BertForSequenceClassification(transformers.BertConfig(**cfg, num_labels=1)).eval()
    for name, model in (("emb", emb), ("rr", rr)):
        model.save_pretrained(root / name)
        tok.save_pretrained(root / name)
    onnx_dir = root / "onnx"
    export_one("embedder", str(root / "emb"), onnx_dir)
    export_one("reranker", str(root / "rr"), onnx_dir)
    return {"tok": tok, "emb": emb, "rr": rr, "onnx": onnx_dir}


TEXTS = ["the employee shall be entitled to annual leave", "personal data without consent", "contract coercion act section", "paid"]


def torch_embed(tiny, texts):
    with torch.no_grad():
        enc = tiny["tok"](texts, padding=True, return_tensors="pt")
        cls = tiny["emb"](**enc).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=1).numpy()


def test_embedder_fp32_matches_pytorch(tiny):
    got = OnnxEmbedder(tiny["onnx"] / "embedder", "fp32").embed(TEXTS)
    assert np.allclose(got, torch_embed(tiny, TEXTS), atol=1e-5)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)


def test_embedder_int8_stays_close_to_fp32(tiny):
    ref = torch_embed(tiny, TEXTS)
    got = OnnxEmbedder(tiny["onnx"] / "embedder", "int8").embed(TEXTS)
    assert ((ref * got).sum(axis=1) > 0.98).all()


def test_embedder_result_does_not_depend_on_batch_size(tiny):
    e = OnnxEmbedder(tiny["onnx"] / "embedder", "fp32")
    assert np.allclose(e.embed(TEXTS, batch_size=1), e.embed(TEXTS, batch_size=32), atol=1e-5)


def test_reranker_fp32_matches_pytorch(tiny):
    docs = ["annual leave eight days", "data consent", "coercion act", "service months paid entitled"]
    with torch.no_grad():
        enc = tiny["tok"](["employee leave days"] * 4, docs, padding=True, return_tensors="pt")
        ref = torch.sigmoid(tiny["rr"](**enc).logits.reshape(-1)).numpy()
    got = np.array(OnnxReranker(tiny["onnx"] / "reranker", "fp32").score("employee leave days", docs))
    assert np.allclose(got, ref, atol=1e-5)


def test_reranker_handles_empty_and_many_docs(tiny):
    r = OnnxReranker(tiny["onnx"] / "reranker", "int8")
    assert r.score("employee leave", []) == []
    assert len(r.score("employee leave", ["annual leave days"] * 37, batch_size=16)) == 37


def test_inputs_much_longer_than_the_traced_example_still_match(tiny):
    """The graph is traced with a ~12 token example. Longer inputs must not hit baked-in lengths."""
    long_texts = [" ".join(["employee leave days data consent"] * 9), " ".join(["contract coercion"] * 24), "paid"]
    got = OnnxEmbedder(tiny["onnx"] / "embedder", "fp32").embed(long_texts, batch_size=2)
    assert np.allclose(got, torch_embed(tiny, long_texts), atol=1e-5)

    docs = [" ".join(["annual leave eight days service"] * 8), "data consent", " ".join(["coercion act"] * 20)]
    with torch.no_grad():
        enc = tiny["tok"](["employee leave days"] * 3, docs, padding=True, return_tensors="pt")
        ref = torch.sigmoid(tiny["rr"](**enc).logits.reshape(-1)).numpy()
    got = np.array(OnnxReranker(tiny["onnx"] / "reranker", "fp32").score("employee leave days", docs, batch_size=3))
    assert enc["input_ids"].shape[1] > 45  # really longer than the traced example
    assert np.allclose(got, ref, atol=1e-5)
