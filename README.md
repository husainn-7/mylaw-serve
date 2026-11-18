# MyLaw-Serve

Serving and benchmarking layer for the models behind [MyLaw-RAG](https://github.com/husainn-7/mylaw-rag), a question-answering system over three Malaysian Acts.

![CI](https://github.com/husainn-7/mylaw-serve/actions/workflows/ci.yml/badge.svg)

A FastAPI service wraps the two models (a bge-small embedder and a MiniLM cross-encoder reranker) and a benchmark harness measures what happens to **speed, memory and retrieval quality** when the same models are run in different forms: PyTorch fp32, ONNX Runtime, int8, and on a GPU. Every variant is re-checked against the same 79 evaluation questions, so a faster model that retrieves worse shows up in the numbers.

## What I found

- **The service reproduces MyLaw-RAG exactly.** Retrieval metrics differ by `+0.000` on every measure, on CPU and on GPU.
- **A GPU is the big win.** Reranking 30 passages takes about 157 ms on a Colab T4 against about 10.7 s on a 2-core Colab CPU (roughly 68x); a whole query takes 0.18 s against about 9 s.
- **ONNX and int8 gave no clear speedup on the 2-core Colab CPU.** ONNX fp32 retrieves identically to PyTorch but was not faster (and used about twice the memory). int8 looked 15-25% faster at reranking in the micro-benchmark but not end to end, and sessions differ by about 10% on their own, so I do not claim an int8 speedup.
- **Retrieval quality barely moved with int8.** Hard-question Hit@1 and Hit@5 were identical for every variant; MRR moved by 0.004 to 0.029, which is about one question's worth on a 28-question split.
- **Scoring fewer candidates is the lever that works on CPU.** Reranking cost is linear in the number of candidates (about 0.28 s each on this CPU). Easy questions lose almost nothing down to 5 candidates; hard questions pay for it.

## Results

All speed numbers are p50 over repeated runs. Quality is hardware independent: 79 answerable questions, 51 easy and 28 hard (see MyLaw-RAG for how they were built). One hard question is worth 0.036, one easy question 0.020.

### Model variants (Colab CPU, 2 cores, one session)

| Variant | hard Hit@1 | hard Hit@5 | hard MRR | easy MRR | rerank 30 passages | whole query | RAM |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 (baseline) | 0.357 | 0.679 | 0.510 | 0.904 | 10.7 s | 9.2 s | 1.0 GB |
| ONNX fp32 | 0.357 | 0.679 | 0.510 | 0.904 | 11.8 s | 12.0 s | 2.2 GB |
| ONNX, int8 reranker only | 0.357 | 0.679 | 0.506 | 0.889 | 9.4 s | 9.8 s | 2.1 GB |
| ONNX, int8 both models | 0.357 | 0.679 | 0.481 | 0.906 | 9.1 s | 9.8 s | 2.0 GB |

For scale, PyTorch fp32 on a Colab T4 GPU (separate session): hard MRR 0.510, easy MRR 0.904, rerank 30 passages 157 ms, whole query 176 ms, 1.5 GB RAM.

### Candidates sent to the reranker (PyTorch fp32, Colab CPU, one session)

| Candidates | whole query | overall MRR | easy MRR | hard MRR | hard Hit@1 |
|---|---|---|---|---|---|
| 5 | 1.5 s | 0.705 | 0.901 | 0.348 | 0.214 |
| 10 | 3.2 s | 0.724 | 0.897 | 0.410 | 0.214 |
| 15 | 5.2 s | 0.737 | 0.886 | 0.465 | 0.286 |
| 20 | 6.2 s | 0.753 | 0.904 | 0.478 | 0.321 |
| 30 (default) | 8.3 s | 0.764 | 0.904 | 0.510 | 0.357 |

### GPU behaviour (Colab T4, fp32)

- Embedding cost per item falls from 10 ms (one query) to 0.62 ms (batch of 32), so batching has a large effect on the embedder.
- Reranking costs about 5 ms per passage whether 10 or 30 are scored.
- Under a live load test (1 to 8 concurrent clients), throughput stayed flat at about 5 requests per second while latency grew from 183 ms to 1.5 s: requests queue behind each other.

## Caveats

- **Each configuration was measured once or twice, on shared Colab hardware.** The same PyTorch CPU configuration gave 10.1 s and 9.2 s per query in two sessions, so differences under about 10% between variants are not distinguishable. Speed numbers are only compared within one session.
- **No CPU throughput numbers.** A query takes about 10 s on the CPU, so the 15-second load test completed one request or fewer. The load test is only meaningful on the GPU.
- **79 evaluation questions.** The hard split has 28, so small quality differences are noise.
- Not investigated: why ONNX used twice the memory of PyTorch, why ONNX fp32 was not faster, and whether this Colab CPU supports fast int8 instructions. The last one is a guess until checked.

## Service

```
POST /search   {"query": "...", "top_sections": 3}   dense search, rerank, grouped by section, with per-stage timings
POST /embed    {"texts": ["..."]}                    embeddings (vectors optional)
POST /rerank   {"query": "...", "docs": ["..."]}     cross-encoder scores
GET  /healthz                                        which engines are loaded
```

Engines are selectable with `ENGINE=torch|onnx|onnx-int8-rerank|onnx-int8` (`DEVICE=cpu|cuda` for torch). Each engine builds its own index, because quantizing the embedder changes the corpus embeddings.

## Run it

```bash
pip install -r requirements-dev.txt          # add requirements-onnx.txt for the ONNX variants
python -m mlserve.onnx_export                # converts both models to ONNX and makes int8 copies

uvicorn mlserve.api:app --port 8000          # service, interactive docs at /docs

python bench/quality.py --engine torch                # retrieval quality; add --pool N to rerank fewer candidates
python bench/micro.py --engine onnx --device cpu      # per-model latency by batch size, memory
python bench/loadtest.py --concurrency 1,4,8          # live load test (needs the server running)
python bench/report.py --markdown                     # every variant side by side

pytest -q                                    # 43 tests; they use fake engines, no model downloads
```

`notebooks/mylaw_serve_cpu_variants.ipynb` runs all variants in one Colab CPU session. Raw results for every run are in `bench/results/`.

## Testing

43 tests cover the service, the API, the benchmark scripts (including a live-server load test), and the metrics. The ONNX tests build tiny random BERT models and check that the ONNX outputs match PyTorch to 1e-5, including on inputs much longer than the example the exporter was traced with. They are skipped automatically where the ONNX libraries are not installed.

## Planned

- Dynamic request batching, to test whether grouping concurrent requests raises throughput on the GPU.
- fp16 on the GPU.
- Adaptive candidate count: a small pool when the dense search is confident, a larger one when it is not. This is a hypothesis; the easy-question results suggest it could work.
- A CPU load test with a long enough window to be meaningful.
