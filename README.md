# llm-inference-lab

[![CI](https://github.com/allenwu-blip/llm-inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/allenwu-blip/llm-inference-lab/actions/workflows/ci.yml)

A small, **reproducible** harness for measuring and optimizing LLM inference. Serve a model, measure it honestly (latency, throughput, memory), sweep an optimization, and get a before → after table you can trust and re-run.

Built to answer a concrete question — *what does quantization actually buy you?* — with real numbers instead of folklore.

## Results — Qwen2.5-0.5B-Instruct on an Apple M3 (Metal)

Quantization sweep, 3 prompts × 128 tokens each, `seed=7`. Every variant measured in its **own process** so the memory number is that variant alone.

| variant | decode tok/s | TTFT ms | peak RSS MB | model on disk MB |
|---|---|---|---|---|
| fp16   | 74.8  | 35.8 | 1336 | 1208 |
| q8_0   | 121.6 | 26.4 | 809  | 644  |
| q4_k_m | 130.7 | 25.9 | 657  | 469  |

![quantization sweep: speed vs memory](results/qwen0.5b.png)

**Reading it:**
- **q4_k_m decodes 1.75× faster than fp16 while using ~half the RAM** (657 vs 1336 MB) and 2.6× less disk.
- **q8_0 is the sweet spot here**: within 8% of q4's speed, near-lossless quality, 809 MB.
- Why: token *decode* is memory-bandwidth-bound, so shrinking the weights speeds it up. On a 0.5B model the gaps are already clear; they widen on larger models (see Roadmap).

## Cross-hardware: Apple M3 (Metal) vs NVIDIA T4 (CUDA)

Same model, same config, same harness — only the backend build changes (Metal → CUDA). The T4 run is on a free Google Colab instance (see [`docs/running-on-cuda.md`](docs/running-on-cuda.md)).

**NVIDIA T4 (CUDA):**

| variant | decode tok/s | TTFT ms | peak RSS MB\* | model on disk MB |
|---|---|---|---|---|
| fp16   | 136.5 | 117.8 | 1586 | 1208 |
| q8_0   | 223.2 | 52.8  | 1022 | 644  |
| q4_k_m | 256.7 | 50.5  | 846  | 469  |

**What running on both reveals** — and you can't see it from one machine:

- **Throughput → the GPU wins.** The T4 decodes ~1.8–2× faster than the M3 (q4_k_m: 257 vs 131 tok/s). Raw generation is where the datacenter GPU pulls ahead.
- **Latency → the M3 wins.** Time-to-first-token is *lower* on the M3 (q4_k_m: 25.9 vs 50.5 ms; fp16: 3× lower). A 0.5B model doesn't saturate the T4, so per-request overhead dominates its TTFT, while the M3's unified memory keeps first-token latency low. A real throughput-vs-latency tradeoff.
- **\*Memory isn't comparable as-is.** On the T4, `peak RSS` is *host* memory (CUDA runtime + staging); the weights live in VRAM, which this harness doesn't measure yet. So RSS is only meaningful within a platform. A VRAM probe is the honest next step.
- The quantization tradeoff holds on both platforms: q4_k_m is fastest and smallest everywhere.

## How it measures (the part that matters)

A benchmark is only worth the honesty of its measurement. Two decisions here:

1. **Per-variant process isolation.** A first version ran all variants in one process and reported peak RSS from `getrusage` — which returns the *whole process* lifetime peak, so every variant showed the same 1294 MB. That's a lie. `--isolate` runs each variant in a fresh process, so peak RSS reflects that variant only. The numbers above use it.
2. **Reproducible by construction.** The model, prompts, generation params, and seed live in a config file; dependencies are pinned in `uv.lock`; one command reproduces the table. Nothing is measured by hand.

`decode tok/s` is tokens after the first divided by decode time, `(generated_tokens - 1) / (total_s - ttft_s)` — prefill is excluded so it reflects steady-state generation.

## Architecture

The measurement layer never touches the engine, so the same harness runs on Metal today and NVIDIA/CUDA later by swapping one file.

```
config (YAML) → runner → Backend.generate() → raw timings → metrics → results (JSON) → table + plot
                              ↑
              fake (tests) · llama.cpp (Metal) · [CUDA / vLLM — next]
```

- `backends/base.py` — `Backend` protocol + `GenerationResult`
- `backends/fake.py` — deterministic backend for tests (no model needed)
- `backends/llama_cpp.py` — real engine, streams tokens to capture true TTFT
- `metrics.py` — pure, unit-tested functions (throughput, TTFT, output similarity, aggregation)
- `runner.py` — orchestration, per-process isolation, JSON persistence
- `report.py` — markdown table + matplotlib plot

## Run it

```bash
uv sync --extra dev
uv run pytest                        # unit tests (deterministic, no model)

# real run: install the Metal engine + a model, then sweep
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python
#   download fp16 / q8_0 / q4_k_m GGUFs into ./models  (Qwen2.5-0.5B-Instruct-GGUF on HF)
uv run python scripts/run.py --config configs/quant-sweep-0.5b.yaml --isolate
```

`--fake` runs the whole pipeline with a synthetic backend if you just want to see it turn.

## Roadmap

- **Per-variant VRAM** — on a GPU the honest memory number is VRAM, not host RSS; add a VRAM probe so the memory column is comparable across hardware.
- **Larger models** — the fp16↔q4 gap grows with model size (more bandwidth-bound); the T4's 16 GB VRAM makes 1.5B / 3B easy next.
- **GPU-only experiments** — continuous batching throughput, speculative decoding.
- **Quality dimension** — outputs are captured per variant and an `output_similarity` metric exists; a full perplexity eval is the next measurement to add.

## Tests

`uv run pytest` — 14 unit tests (pure metrics, config, fake backend, runner, report) plus a llama.cpp smoke test that **skips** unless `LLAMA_TEST_MODEL` points at a GGUF (it never fabricates a pass).

MIT.
