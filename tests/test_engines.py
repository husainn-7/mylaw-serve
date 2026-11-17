import json

import pytest

from mlserve.engines import ENGINES, combine_names, load_engines
from mlserve.onnx_engines import OnnxEmbedder, OnnxReranker


def test_combine_names():
    assert combine_names("torch-fp32-cuda", "torch-fp32-cuda") == "torch-fp32-cuda"
    assert combine_names("onnx-fp32-cpu", "onnx-int8-cpu") == "onnx-fp32-cpu_onnx-int8-cpu"


def test_unknown_engine_lists_the_available_ones():
    with pytest.raises(ValueError) as e:
        load_engines("nope")
    for name in ENGINES:
        assert name in str(e.value)


def test_fake_engine_loads():
    embedder, reranker = load_engines("fake")
    assert embedder.name == reranker.name == "fake"


def test_missing_onnx_files_explain_how_to_create_them(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        OnnxEmbedder(tmp_path / "embedder")
    assert "onnx_export" in str(e.value)
    with pytest.raises(FileNotFoundError):
        OnnxReranker(tmp_path / "reranker", "int8")


def test_load_engines_onnx_reports_missing_models(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_engines("onnx-int8-rerank")
