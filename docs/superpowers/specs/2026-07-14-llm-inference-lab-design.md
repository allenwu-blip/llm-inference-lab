# llm-inference-lab — Design Spec

**Date:** 2026-07-14
**Status:** Design approved (brainstorming) → next: implementation plan

## Goal

A reproducible harness that benchmarks and optimizes LLM inference on commodity hardware. It serves a small language model, measures inference performance (latency, throughput, memory), applies optimization techniques one at a time, and records **before → after** results with analysis. Portfolio artifact demonstrating AI-infrastructure / performance-engineering skill, grounded in rigorous measurement.

## Why

- Career direction: ML/AI infrastructure — "build AND make it run fast."
- Reuses the author's measurement / evaluation discipline.
- Runs locally at $0 on Apple Silicon (M3, 8 GB); the same harness is portable to NVIDIA later.

## Non-goals (YAGNI)

- No model training or fine-tuning.
- No paid cloud in Phase 1.
- No continuous batching or speculative decoding in Phase 1 (Phase 2 / stretch).
- Not a serving product — a benchmark + analysis harness.

## Architecture

Four decoupled units, each independently testable:

1. **Config** — declarative run configuration (model, quantization, prompt set, generation params, backend choice).
2. **Backend (abstraction)** — a `Backend` protocol: `load(model, quant)` and `generate(prompt, params) -> (tokens, timing)`. First implementation: llama.cpp via `llama-cpp-python` (Metal on M3; the same GGUF model runs CUDA later). Future implementations: CUDA build of the same, or vLLM (GPU only).
3. **Metrics** — pure functions computing TTFT, inter-token latency, decode throughput (tok/s), and peak memory (RSS) from raw timing/token data. Deterministic → unit-tested.
4. **Runner + Report** — orchestrates runs across a config matrix, persists raw results (JSON), and renders a results table + plot (matplotlib) + a markdown writeup.

Data flow: `config → runner → backend.generate() (per prompt) → raw timings → metrics → results store → report`.

The measurement layer never touches the engine directly, so swapping Metal → CUDA → vLLM changes only the backend unit.

## Phase 1 (local M3, now)

- **Model:** default ~1.5B (e.g. Qwen2.5-1.5B-Instruct, GGUF), chosen so fp16 (~3 GB) fits in 8 GB for a full quantization sweep. Configurable.
- **Experiment:** quantization sweep — fp16 / int8 (Q8_0) / int4 (Q4_K_M).
- **Metrics per config:** TTFT (ms), decode throughput (tok/s), peak RSS (MB), on-disk model size (MB).
- **Quality guard:** a fixed prompt set + a deterministic quality measure (perplexity on a held text, or output-similarity) so the report shows the *quality cost* of quantization, not just speed.
- **Outputs:** `results/*.json` (raw), a markdown results table, one plot (speed & memory vs quantization), and a README writeup: methodology + table + analysis (e.g. "int4 ≈ 4× smaller, X% faster decode, Y quality delta — because decode is memory-bandwidth-bound").

## Phase 2 (UMich NVIDIA GPU, later)

- Same harness, CUDA backend (llama.cpp CUDA build; optional vLLM backend).
- GPU-specific experiments: batch size vs aggregate throughput; larger model.
- **Cross-hardware comparison:** M3-Metal vs NVIDIA-CUDA on the same model and metrics.

## Tech choices

- Python (managed with `uv`), `llama-cpp-python` (Metal build now), `matplotlib` for plots, `pytest` for tests.
- If any LLM-judged quality step is added later, use Anthropic (standing preference). Phase 1's quality guard is local and deterministic — no external API needed.

## Testing

- **Unit (deterministic):** metrics computations (TTFT / throughput / aggregation), config parsing + validation, backend-interface contract via a fake backend.
- **Smoke:** one tiny real generation confirming backend wiring (skipped if the model file is absent).

## Deliverable

Public repo `llm-inference-lab`: harness code, tests, `results/`, README writeup. No internal/business content → safe to publish directly (unlike agibridge).

## Hardware constraint

Apple M3, 8 GB RAM → target ≤ 3B models; default 1.5B so the full fp16 → int4 sweep fits without OOM.
