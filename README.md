# MyLaw-Serve

Serving and benchmarking layer for the models behind [MyLaw-RAG](https://github.com/husainn-7/mylaw-rag):
a FastAPI service plus a benchmark harness that measures **latency, throughput, memory and retrieval quality**
for the same models in different forms (PyTorch fp32, ONNX Runtime, int8, GPU).

Status: work in progress. Day 1 = PyTorch baseline service and benchmark harness.

## Run it

```bash
pip install -r requirements-dev.txt
# data/chunks.jsonl and data/questions.jsonl come from the MyLaw-RAG repo

uvicorn mlserve.api:app --port 8000        # service: /search /embed /rerank /healthz  (docs at /docs)

python bench/quality.py --engine torch     # retrieval quality, compared with MyLaw-RAG
python bench/micro.py --engine torch       # per-model latency by batch size, memory
python bench/loadtest.py --concurrency 1,4,8 --duration 15   # needs the server running

pytest -q                                  # tests use fake engines, no model downloads
```
