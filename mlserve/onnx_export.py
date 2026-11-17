"""Convert the two models to ONNX and make int8 copies.

    python -m mlserve.onnx_export                 # writes models/onnx/embedder and models/onnx/reranker
"""
import argparse
import os
from pathlib import Path

from mlserve.engines import EMBED_MODEL, RERANK_MODEL

INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]


def _wrapper(model):
    import torch

    class Wrapper(torch.nn.Module):
        """Return a plain tensor (first output) instead of a Hugging Face output object."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False
            )[0]

    return Wrapper(model).eval()


def _export(model, tokenizer, pair, out_file, output_name, opset=17):
    import torch

    if pair:
        enc = tokenizer(
            ["how many days of annual leave", "what is coercion"],
            ["An employee shall be entitled to paid annual leave of eight days", "Coercion is the committing of any act forbidden by law"],
            padding=True, return_tensors="pt",
        )
    else:
        enc = tokenizer(["hello world", "a somewhat longer example sentence used for tracing the graph"], padding=True, return_tensors="pt")
    args = tuple(enc[n] for n in INPUT_NAMES)
    dynamic_axes = {n: {0: "batch", 1: "seq"} for n in INPUT_NAMES}
    dynamic_axes[output_name] = {0: "batch", 1: "seq"} if not pair else {0: "batch"}
    kwargs = dict(input_names=INPUT_NAMES, output_names=[output_name], dynamic_axes=dynamic_axes, opset_version=opset)
    with torch.no_grad():
        try:
            torch.onnx.export(_wrapper(model), args, str(out_file), dynamo=False, **kwargs)
        except TypeError:  # older PyTorch has no 'dynamo' argument and always uses the classic exporter
            torch.onnx.export(_wrapper(model), args, str(out_file), **kwargs)


def quantize_int8(src, dst):
    """Dynamic int8 quantization of the weights (activations are quantized on the fly at run time)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def export_one(kind, model_name, out_dir):
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out_dir) / kind
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(out)

    if kind == "embedder":
        model = AutoModel.from_pretrained(model_name, attn_implementation="eager").eval()
        _export(model, tokenizer, pair=False, out_file=out / "model.onnx", output_name="last_hidden_state")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, attn_implementation="eager").eval()
        _export(model, tokenizer, pair=True, out_file=out / "model.onnx", output_name="logits")

    quantize_int8(out / "model.onnx", out / "model_int8.onnx")
    sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in out.glob("*.onnx")}
    print(f"{kind}: {sizes} MB")
    return out


def export_all(out_dir="models/onnx", embed_model=EMBED_MODEL, rerank_model=RERANK_MODEL):
    export_one("embedder", embed_model, out_dir)
    export_one("reranker", rerank_model, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("MODELS_DIR", "models/onnx"))
    export_all(ap.parse_args().out)
