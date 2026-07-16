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
- Why: token *decode* is memory-bandwidth-bound, so shrinking the weights speeds it up. On a 0.5B model the gaps are already clear; they widen on larger models (shown below on a 3B run).

## Cross-hardware: Apple M3 (Metal) vs NVIDIA T4 & V100 (CUDA)

Same 0.5B model, same config, same harness — only the backend build changes. The T4 runs on a free Google Colab instance ([`docs/running-on-cuda.md`](docs/running-on-cuda.md)); the V100 runs on the University of Michigan **Great Lakes HPC cluster** via Slurm ([`docs/running-on-greatlakes.md`](docs/running-on-greatlakes.md)).

**decode tok/s** (higher is better):

| variant | M3 Metal | T4 CUDA | V100 CUDA |
|---|---|---|---|
| fp16   | 74.8  | 136.5 | 274.1 |
| q8_0   | 121.6 | 223.2 | 330.8 |
| q4_k_m | 130.7 | 256.7 | 328.2 |

**TTFT ms** (lower is better):

| variant | M3 Metal | T4 CUDA | V100 CUDA |
|---|---|---|---|
| fp16   | 35.8 | 117.8 | 51.4 |
| q8_0   | 26.4 | 52.8  | 32.3 |
| q4_k_m | 25.9 | 50.5  | 35.7 |

**Three things you only see across hardware:**

- **Throughput scales with the GPU.** V100 > T4 > M3 (fp16: 274 vs 137 vs 75 tok/s). Raw generation is where the bigger datacenter GPU pulls ahead.
- **Latency doesn't.** The M3 has the *lowest* TTFT (fp16: 36 ms, vs V100 51, T4 118). A 0.5B model doesn't saturate the GPUs, so per-request overhead dominates their first-token latency while the M3's unified memory keeps it low — a real throughput-vs-latency tradeoff.
- **The quantization win flattens on the fast GPU.** On the M3 and T4, `q4_k_m` is the fastest variant; on the V100, `q8_0` actually edges it (330.8 vs 328.2 tok/s). A model this small can't saturate a V100's memory bandwidth, so int4's extra dequantization cost stops paying off — which points straight to the next experiment: a *larger* model.
- Memory (`peak RSS`) is host memory on the GPUs; the weights live in VRAM (not yet measured), so it's only comparable within a platform.

## Model size changes the answer — Qwen2.5-3B on the V100

Re-running the sweep on a 6× larger model (3B) on the same V100 brings the bandwidth bottleneck back, and with it the quantization advantage:

| variant | decode tok/s | TTFT ms | peak RSS MB | vs fp16 |
|---|---|---|---|---|
| fp16   | 96.2  | 22.4 | 6878 | — |
| q8_0   | 136.0 | 22.6 | 3841 | 1.41× |
| q4_k_m | 159.5 | 25.1 | 2399 | **1.66× faster, ~⅓ the RAM** |

The V100's quantization speedup grew from **1.2× (0.5B) to 1.66× (3B)**, and the near-tie `q8_0 ≈ q4_k_m` from the small model resolved into a clean `q4 > q8 > fp16` ordering. The takeaway, backed by data across two model sizes and three devices:

> **Quantization pays off in proportion to how memory-bandwidth-bound decode is** — which rises with model size and falls as the hardware overpowers the model. It isn't "int4 is always faster"; it's "int4 is faster when the weights are the bottleneck."

*(fp16 3B ships as a split GGUF; `peak RSS` reflects the full ~6.4 GB loaded, while `model on disk` in the raw results counts only the first shard.)*

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
              fake (tests) · llama.cpp (Metal + CUDA) · [vLLM — next]
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

For NVIDIA/CUDA — Colab's T4 or Great Lakes' V100 — see [`docs/running-on-cuda.md`](docs/running-on-cuda.md) and [`docs/running-on-greatlakes.md`](docs/running-on-greatlakes.md).

## Roadmap

- **Per-variant VRAM** — on a GPU the honest memory number is VRAM, not host RSS; add a VRAM probe so the memory column is comparable across hardware.
- **GPU-only experiments** — continuous batching throughput and speculative decoding, where a big GPU's parallelism really shows.
- **Quality dimension** — outputs are captured per variant and an `output_similarity` metric exists; a full perplexity eval is the next measurement to add.
- **Bigger still** — 7B+ on the V100's 16 GB VRAM, to see how far the quantization advantage widens.

## Tests

`uv run pytest` — 14 unit tests (pure metrics, config, fake backend, runner, report) plus a llama.cpp smoke test that **skips** unless `LLAMA_TEST_MODEL` points at a GGUF (it never fabricates a pass).

MIT.
