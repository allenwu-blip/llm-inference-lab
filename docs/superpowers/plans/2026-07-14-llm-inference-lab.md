# llm-inference-lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible harness that benchmarks a small LLM's inference (latency, throughput, memory, quality) across a quantization sweep on Apple Silicon, producing a results table + plot + writeup.

**Architecture:** Four decoupled units — `config` (declarative runs), `Backend` protocol (engine behind an interface; fake + llama.cpp impls), `metrics` (pure functions over raw timings), `runner`+`report` (orchestrate + render). The measurement layer never touches the engine, so a CUDA/vLLM backend can be added later without changing metrics.

**Tech Stack:** Python 3.12+ (managed with `uv`), `llama-cpp-python` (Metal build), `matplotlib`, `pytest`, `PyYAML`.

## Global Constraints

- Package name: `inference_lab` under `src/`. Import as `from inference_lab import ...`.
- Python managed with `uv`; run tests with `uv run pytest`.
- Pure metric functions must be deterministic and have zero I/O — unit-tested without any model.
- Real-model code (`LlamaCppBackend`) is smoke-tested only, and the smoke test is **skipped** when the model file is absent (never fabricate a pass).
- Default model: a ~1.5B instruct model in GGUF (fp16 fits in 8 GB); quant labels used verbatim: `fp16`, `q8_0`, `q4_k_m`.
- No secrets, no external API calls in Phase 1. Repo is public-safe.
- `decode_throughput` definition (verbatim): tokens after the first, divided by decode time = `(generated_tokens - 1) / (total_s - ttft_s)`.

---

## File Structure

- `pyproject.toml` — uv project + deps
- `src/inference_lab/__init__.py` — package marker
- `src/inference_lab/backends/base.py` — `GenerationResult` dataclass + `Backend` protocol
- `src/inference_lab/backends/fake.py` — `FakeBackend` (deterministic, for tests)
- `src/inference_lab/backends/llama_cpp.py` — `LlamaCppBackend` (real, Metal)
- `src/inference_lab/metrics.py` — pure functions: `decode_throughput`, `ttft_ms`, `output_similarity`, `aggregate` → `RunMetrics`
- `src/inference_lab/config.py` — `ExperimentConfig` + `RunConfig` + `load_experiment` + validation
- `src/inference_lab/runner.py` — `RunResult` + `run_experiment` + JSON persistence
- `src/inference_lab/report.py` — `render_table`, `render_plot`
- `scripts/run.py` — CLI wiring
- `configs/quant-sweep.yaml` — Phase 1 experiment
- `tests/…` — one test module per unit

---

### Task 1: Scaffold + core types

**Files:**
- Create: `pyproject.toml`, `src/inference_lab/__init__.py`, `src/inference_lab/backends/__init__.py`, `src/inference_lab/backends/base.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Produces: `GenerationResult(prompt: str, output_text: str, prompt_tokens: int, generated_tokens: int, ttft_s: float, total_s: float)` dataclass; `Backend` protocol with `name: str`, `load() -> None`, `generate(prompt: str, max_tokens: int, seed: int) -> GenerationResult`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "inference-lab"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pyyaml>=6", "matplotlib>=3.8"]

[project.optional-dependencies]
dev = ["pytest>=8"]
llama = ["llama-cpp-python>=0.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/inference_lab"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package markers**

`src/inference_lab/__init__.py` and `src/inference_lab/backends/__init__.py` — both empty files.

- [ ] **Step 3: Write the failing test** — `tests/test_base.py`

```python
from inference_lab.backends.base import GenerationResult

def test_generation_result_holds_fields():
    r = GenerationResult(
        prompt="hi", output_text="hello there",
        prompt_tokens=1, generated_tokens=2, ttft_s=0.1, total_s=0.3,
    )
    assert r.generated_tokens == 2
    assert r.total_s == 0.3
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_base.py -v`
Expected: FAIL (ModuleNotFoundError: inference_lab.backends.base).

- [ ] **Step 5: Implement `src/inference_lab/backends/base.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationResult:
    prompt: str
    output_text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float   # seconds to first generated token
    total_s: float  # seconds for the whole generation call


class Backend(Protocol):
    name: str
    def load(self) -> None: ...
    def generate(self, prompt: str, max_tokens: int, seed: int) -> GenerationResult: ...
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_base.py -v` → Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/inference_lab tests/test_base.py
git commit -m "feat: project scaffold + GenerationResult/Backend core types"
```

---

### Task 2: metrics (pure functions)

**Files:**
- Create: `src/inference_lab/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `GenerationResult` from Task 1.
- Produces: `decode_throughput(generated_tokens: int, ttft_s: float, total_s: float) -> float`; `ttft_ms(result: GenerationResult) -> float`; `output_similarity(reference: str, candidate: str) -> float` (0..1); `aggregate(results: list[GenerationResult]) -> RunMetrics`. `RunMetrics(n: int, ttft_ms_mean: float, ttft_ms_median: float, decode_tps_mean: float, decode_tps_median: float, generated_tokens_total: int)`.

- [ ] **Step 1: Write failing tests** — `tests/test_metrics.py`

```python
import math
from inference_lab.backends.base import GenerationResult
from inference_lab.metrics import decode_throughput, ttft_ms, output_similarity, aggregate

def test_decode_throughput_basic():
    # 11 tokens, first at 0.5s, done at 1.5s -> 10 tokens over 1.0s decode = 10 tps
    assert decode_throughput(11, 0.5, 1.5) == 10.0

def test_decode_throughput_guards_zero():
    assert decode_throughput(1, 0.5, 0.5) == 0.0   # only one token / no decode window
    assert decode_throughput(0, 0.0, 0.0) == 0.0

def test_ttft_ms():
    r = GenerationResult("p", "o", 1, 5, 0.25, 1.0)
    assert ttft_ms(r) == 250.0

def test_output_similarity_bounds():
    assert output_similarity("hello world", "hello world") == 1.0
    assert output_similarity("abc", "xyz") < 0.5

def test_aggregate_means_and_medians():
    rs = [
        GenerationResult("p", "o", 1, 11, 0.5, 1.5),   # decode 10 tps, ttft 500ms
        GenerationResult("p", "o", 1, 21, 1.0, 3.0),   # decode 10 tps, ttft 1000ms
    ]
    m = aggregate(rs)
    assert m.n == 2
    assert m.ttft_ms_mean == 750.0
    assert math.isclose(m.decode_tps_mean, 10.0)
    assert m.generated_tokens_total == 32
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_metrics.py -v` → Expected: FAIL (no module `metrics`).

- [ ] **Step 3: Implement `src/inference_lab/metrics.py`**

```python
from __future__ import annotations
import statistics
from dataclasses import dataclass
from difflib import SequenceMatcher
from .backends.base import GenerationResult


def decode_throughput(generated_tokens: int, ttft_s: float, total_s: float) -> float:
    decode_time = total_s - ttft_s
    if generated_tokens < 2 or decode_time <= 0:
        return 0.0
    return (generated_tokens - 1) / decode_time


def ttft_ms(result: GenerationResult) -> float:
    return result.ttft_s * 1000.0


def output_similarity(reference: str, candidate: str) -> float:
    return SequenceMatcher(None, reference, candidate).ratio()


@dataclass
class RunMetrics:
    n: int
    ttft_ms_mean: float
    ttft_ms_median: float
    decode_tps_mean: float
    decode_tps_median: float
    generated_tokens_total: int


def aggregate(results: list[GenerationResult]) -> RunMetrics:
    ttfts = [ttft_ms(r) for r in results]
    tps = [decode_throughput(r.generated_tokens, r.ttft_s, r.total_s) for r in results]
    return RunMetrics(
        n=len(results),
        ttft_ms_mean=statistics.mean(ttfts),
        ttft_ms_median=statistics.median(ttfts),
        decode_tps_mean=statistics.mean(tps),
        decode_tps_median=statistics.median(tps),
        generated_tokens_total=sum(r.generated_tokens for r in results),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_metrics.py -v` → Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/inference_lab/metrics.py tests/test_metrics.py
git commit -m "feat: pure metrics (throughput, ttft, similarity, aggregate)"
```

---

### Task 3: config

**Files:**
- Create: `src/inference_lab/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `RunConfig(name: str, model_path: str, quant: str, prompts: list[str], max_tokens: int, seed: int)`; `ExperimentConfig(name: str, prompts: list[str], max_tokens: int, seed: int, variants: dict[str, str])` (variants = quant_label -> model_path); `load_experiment(path: str) -> ExperimentConfig` (raises `ValueError` on invalid); `ExperimentConfig.expand() -> list[RunConfig]`.

- [ ] **Step 1: Write failing tests** — `tests/test_config.py`

```python
import pytest
from inference_lab.config import ExperimentConfig, load_experiment

def _write(tmp_path, text):
    p = tmp_path / "exp.yaml"; p.write_text(text); return str(p)

def test_load_and_expand(tmp_path):
    path = _write(tmp_path, """
name: sweep
max_tokens: 64
seed: 7
prompts: ["Explain gravity.", "Write a haiku."]
variants:
  fp16: models/m-fp16.gguf
  q4_k_m: models/m-q4.gguf
""")
    exp = load_experiment(path)
    runs = exp.expand()
    assert exp.name == "sweep"
    assert len(runs) == 2
    assert {r.quant for r in runs} == {"fp16", "q4_k_m"}
    assert runs[0].max_tokens == 64 and runs[0].seed == 7
    assert runs[0].prompts == ["Explain gravity.", "Write a haiku."]

def test_invalid_config_raises(tmp_path):
    path = _write(tmp_path, "name: bad\nmax_tokens: 0\nprompts: []\nvariants: {}\nseed: 1\n")
    with pytest.raises(ValueError):
        load_experiment(path)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_config.py -v` → Expected: FAIL (no module `config`).

- [ ] **Step 3: Implement `src/inference_lab/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import yaml


@dataclass
class RunConfig:
    name: str
    model_path: str
    quant: str
    prompts: list[str]
    max_tokens: int
    seed: int


@dataclass
class ExperimentConfig:
    name: str
    prompts: list[str]
    max_tokens: int
    seed: int
    variants: dict[str, str]

    def expand(self) -> list[RunConfig]:
        return [
            RunConfig(
                name=f"{self.name}:{quant}", model_path=path, quant=quant,
                prompts=self.prompts, max_tokens=self.max_tokens, seed=self.seed,
            )
            for quant, path in self.variants.items()
        ]


def load_experiment(path: str) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    exp = ExperimentConfig(
        name=data.get("name", "experiment"),
        prompts=list(data.get("prompts", [])),
        max_tokens=int(data.get("max_tokens", 0)),
        seed=int(data.get("seed", 0)),
        variants=dict(data.get("variants", {})),
    )
    if not exp.prompts:
        raise ValueError("config: prompts must be non-empty")
    if exp.max_tokens <= 0:
        raise ValueError("config: max_tokens must be > 0")
    if not exp.variants:
        raise ValueError("config: variants must be non-empty")
    return exp
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -v` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inference_lab/config.py tests/test_config.py
git commit -m "feat: experiment config load/validate/expand"
```

---

### Task 4: FakeBackend

**Files:**
- Create: `src/inference_lab/backends/fake.py`
- Test: `tests/test_fake_backend.py`

**Interfaces:**
- Consumes: `GenerationResult`, `Backend` (Task 1).
- Produces: `FakeBackend(ttft_s: float = 0.05, per_token_s: float = 0.01, name: str = "fake")` implementing `Backend`; deterministic `generate` returns `generated_tokens == max_tokens`, `ttft_s == ttft_s`, `total_s == ttft_s + per_token_s * max_tokens`, `output_text` derived from prompt+seed.

- [ ] **Step 1: Write failing test** — `tests/test_fake_backend.py`

```python
from inference_lab.backends.fake import FakeBackend

def test_fake_backend_is_deterministic():
    b = FakeBackend(ttft_s=0.05, per_token_s=0.01)
    b.load()
    r1 = b.generate("hello", max_tokens=10, seed=1)
    r2 = b.generate("hello", max_tokens=10, seed=1)
    assert r1.generated_tokens == 10
    assert r1.ttft_s == 0.05
    assert abs(r1.total_s - (0.05 + 0.01 * 10)) < 1e-9
    assert r1.output_text == r2.output_text  # deterministic
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_fake_backend.py -v` → Expected: FAIL.

- [ ] **Step 3: Implement `src/inference_lab/backends/fake.py`**

```python
from __future__ import annotations
from .base import GenerationResult


class FakeBackend:
    def __init__(self, ttft_s: float = 0.05, per_token_s: float = 0.01, name: str = "fake"):
        self.ttft_s = ttft_s
        self.per_token_s = per_token_s
        self.name = name

    def load(self) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int, seed: int) -> GenerationResult:
        total_s = self.ttft_s + self.per_token_s * max_tokens
        text = f"[{seed}] " + " ".join(prompt.split()[:3]) * max(1, max_tokens // 4)
        return GenerationResult(
            prompt=prompt, output_text=text,
            prompt_tokens=len(prompt.split()), generated_tokens=max_tokens,
            ttft_s=self.ttft_s, total_s=total_s,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_fake_backend.py -v` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inference_lab/backends/fake.py tests/test_fake_backend.py
git commit -m "feat: deterministic FakeBackend for harness tests"
```

---

### Task 5: runner

**Files:**
- Create: `src/inference_lab/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 3), `Backend`/`GenerationResult` (Task 1), `aggregate`/`RunMetrics` (Task 2).
- Produces: `RunResult(variant: str, metrics: RunMetrics, peak_rss_mb: float, model_size_mb: float, outputs: list[str])`; `run_experiment(runs: list[RunConfig], backend_factory: Callable[[RunConfig], Backend]) -> list[RunResult]`; `save_results(results: list[RunResult], path: str) -> None` (writes JSON).

- [ ] **Step 1: Write failing test** — `tests/test_runner.py`

```python
import json
from inference_lab.config import RunConfig
from inference_lab.backends.fake import FakeBackend
from inference_lab.runner import run_experiment, save_results, RunResult

def _run(quant, per_token_s):
    return RunConfig(name=f"t:{quant}", model_path="none", quant=quant,
                     prompts=["a b c", "d e f"], max_tokens=8, seed=1)

def test_run_experiment_aggregates_per_variant():
    runs = [_run("fp16", 0.02), _run("q4_k_m", 0.01)]
    results = run_experiment(runs, backend_factory=lambda rc: FakeBackend())
    assert [r.variant for r in results] == ["fp16", "q4_k_m"]
    assert results[0].metrics.n == 2               # two prompts
    assert results[0].peak_rss_mb > 0
    assert len(results[0].outputs) == 2

def test_save_results_writes_json(tmp_path):
    runs = [_run("fp16", 0.02)]
    results = run_experiment(runs, backend_factory=lambda rc: FakeBackend())
    out = tmp_path / "r.json"
    save_results(results, str(out))
    data = json.loads(out.read_text())
    assert data[0]["variant"] == "fp16"
    assert "decode_tps_mean" in data[0]["metrics"]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_runner.py -v` → Expected: FAIL.

- [ ] **Step 3: Implement `src/inference_lab/runner.py`**

```python
from __future__ import annotations
import json, os, resource
from dataclasses import dataclass, asdict
from typing import Callable
from .backends.base import Backend
from .config import RunConfig
from .metrics import aggregate, RunMetrics


@dataclass
class RunResult:
    variant: str
    metrics: RunMetrics
    peak_rss_mb: float
    model_size_mb: float
    outputs: list[str]


def _peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KB. Normalize to MB.
    return (kb / (1024 * 1024)) if kb > 10_000_000 else (kb / 1024)


def _model_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def run_experiment(
    runs: list[RunConfig],
    backend_factory: Callable[[RunConfig], Backend],
) -> list[RunResult]:
    results: list[RunResult] = []
    for rc in runs:
        backend = backend_factory(rc)
        backend.load()
        gens = [backend.generate(p, rc.max_tokens, rc.seed) for p in rc.prompts]
        results.append(RunResult(
            variant=rc.quant,
            metrics=aggregate(gens),
            peak_rss_mb=_peak_rss_mb(),
            model_size_mb=_model_size_mb(rc.model_path),
            outputs=[g.output_text for g in gens],
        ))
    return results


def save_results(results: list[RunResult], path: str) -> None:
    payload = [
        {
            "variant": r.variant,
            "metrics": asdict(r.metrics),
            "peak_rss_mb": r.peak_rss_mb,
            "model_size_mb": r.model_size_mb,
            "outputs": r.outputs,
        }
        for r in results
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_runner.py -v` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inference_lab/runner.py tests/test_runner.py
git commit -m "feat: runner orchestrates variants, aggregates, persists JSON"
```

---

### Task 6: report

**Files:**
- Create: `src/inference_lab/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `RunResult` (Task 5).
- Produces: `render_table(results: list[RunResult]) -> str` (markdown); `render_plot(results: list[RunResult], out_path: str) -> None` (writes a PNG).

- [ ] **Step 1: Write failing test** — `tests/test_report.py`

```python
import os
from inference_lab.metrics import RunMetrics
from inference_lab.runner import RunResult
from inference_lab.report import render_table, render_plot

def _res(v, tps, rss, size):
    return RunResult(variant=v, metrics=RunMetrics(2, 500.0, 500.0, tps, tps, 20),
                     peak_rss_mb=rss, model_size_mb=size, outputs=["x", "y"])

def test_render_table_has_rows_and_headers():
    md = render_table([_res("fp16", 8.0, 3000, 3000), _res("q4_k_m", 12.0, 1200, 900)])
    assert "| variant |" in md
    assert "fp16" in md and "q4_k_m" in md
    assert "12.0" in md  # decode tps rendered

def test_render_plot_writes_file(tmp_path):
    out = tmp_path / "plot.png"
    render_plot([_res("fp16", 8.0, 3000, 3000), _res("q4_k_m", 12.0, 1200, 900)], str(out))
    assert os.path.exists(out) and os.path.getsize(out) > 0
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_report.py -v` → Expected: FAIL.

- [ ] **Step 3: Implement `src/inference_lab/report.py`**

```python
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # headless, no display
import matplotlib.pyplot as plt
from .runner import RunResult


def render_table(results: list[RunResult]) -> str:
    header = "| variant | decode tok/s | TTFT ms | peak RSS MB | model MB |"
    sep = "|---|---|---|---|---|"
    rows = [
        f"| {r.variant} | {r.metrics.decode_tps_mean:.1f} | "
        f"{r.metrics.ttft_ms_mean:.1f} | {r.peak_rss_mb:.0f} | {r.model_size_mb:.0f} |"
        for r in results
    ]
    return "\n".join([header, sep, *rows])


def render_plot(results: list[RunResult], out_path: str) -> None:
    variants = [r.variant for r in results]
    tps = [r.metrics.decode_tps_mean for r in results]
    rss = [r.peak_rss_mb for r in results]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(variants, tps, color="#2E86AB", label="decode tok/s")
    ax1.set_ylabel("decode tok/s")
    ax2 = ax1.twinx()
    ax2.plot(variants, rss, color="#C0392B", marker="o", label="peak RSS MB")
    ax2.set_ylabel("peak RSS (MB)")
    ax1.set_title("Quantization sweep: speed vs memory")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_report.py -v` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inference_lab/report.py tests/test_report.py
git commit -m "feat: markdown results table + speed/memory plot"
```

---

### Task 7: LlamaCppBackend + CLI + config + smoke

**Files:**
- Create: `src/inference_lab/backends/llama_cpp.py`, `scripts/run.py`, `configs/quant-sweep.yaml`
- Test: `tests/test_llama_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `LlamaCppBackend(model_path: str, n_ctx: int = 2048, name: str = "llama.cpp")` implementing `Backend`; CLI `python scripts/run.py --config <path> [--out results/run.json] [--plot results/plot.png] [--fake]`.

- [ ] **Step 1: Write the smoke test (skips without a model)** — `tests/test_llama_smoke.py`

```python
import os, pytest

MODEL = os.environ.get("LLAMA_TEST_MODEL")

@pytest.mark.skipif(not MODEL or not os.path.exists(MODEL),
                    reason="set LLAMA_TEST_MODEL to a small GGUF to run")
def test_llama_backend_generates():
    from inference_lab.backends.llama_cpp import LlamaCppBackend
    b = LlamaCppBackend(MODEL, n_ctx=512)
    b.load()
    r = b.generate("Say hi in one word.", max_tokens=8, seed=1)
    assert r.generated_tokens >= 1
    assert r.total_s >= r.ttft_s > 0
    assert isinstance(r.output_text, str) and len(r.output_text) > 0
```

- [ ] **Step 2: Run to verify it is collected and SKIPPED**

Run: `uv run pytest tests/test_llama_smoke.py -v`
Expected: 1 skipped (reason: set LLAMA_TEST_MODEL). This is the correct pass state without a model.

- [ ] **Step 3: Implement `src/inference_lab/backends/llama_cpp.py`**

```python
from __future__ import annotations
import time
from .base import GenerationResult


class LlamaCppBackend:
    def __init__(self, model_path: str, n_ctx: int = 2048, name: str = "llama.cpp"):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.name = name
        self._llm = None

    def load(self) -> None:
        from llama_cpp import Llama  # imported lazily so unit tests don't need it
        self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx,
                          n_gpu_layers=-1, verbose=False)

    def generate(self, prompt: str, max_tokens: int, seed: int) -> GenerationResult:
        assert self._llm is not None, "call load() first"
        start = time.perf_counter()
        first_t: float | None = None
        pieces: list[str] = []
        n = 0
        stream = self._llm(prompt, max_tokens=max_tokens, seed=seed, stream=True)
        for chunk in stream:
            if first_t is None:
                first_t = time.perf_counter()
            pieces.append(chunk["choices"][0]["text"])
            n += 1
        total_s = time.perf_counter() - start
        ttft_s = (first_t - start) if first_t is not None else total_s
        prompt_tokens = len(self._llm.tokenize(prompt.encode()))
        return GenerationResult(
            prompt=prompt, output_text="".join(pieces),
            prompt_tokens=prompt_tokens, generated_tokens=n,
            ttft_s=ttft_s, total_s=total_s,
        )
```

- [ ] **Step 4: Create `configs/quant-sweep.yaml`**

```yaml
name: quant-sweep
max_tokens: 128
seed: 7
prompts:
  - "Explain what a KV cache is in one paragraph."
  - "Write a haiku about the ocean."
  - "List three uses of a hash map."
variants:
  # download GGUFs into ./models first (see README); paths are examples
  fp16: models/qwen2.5-1.5b-instruct-fp16.gguf
  q8_0: models/qwen2.5-1.5b-instruct-q8_0.gguf
  q4_k_m: models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

- [ ] **Step 5: Create `scripts/run.py`**

```python
from __future__ import annotations
import argparse, sys
sys.path.insert(0, "src")
from inference_lab.config import load_experiment
from inference_lab.runner import run_experiment, save_results
from inference_lab.report import render_table, render_plot
from inference_lab.backends.fake import FakeBackend
from inference_lab.backends.llama_cpp import LlamaCppBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/run.json")
    ap.add_argument("--plot", default="results/plot.png")
    ap.add_argument("--fake", action="store_true", help="use FakeBackend (no model)")
    args = ap.parse_args()

    runs = load_experiment(args.config).expand()
    if args.fake:
        factory = lambda rc: FakeBackend()
    else:
        factory = lambda rc: LlamaCppBackend(rc.model_path)
    results = run_experiment(runs, backend_factory=factory)
    save_results(results, args.out)
    render_plot(results, args.plot)
    print(render_table(results))
    print(f"\nsaved: {args.out}  plot: {args.plot}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the CLI works end-to-end with the fake backend**

Run: `uv run python scripts/run.py --config configs/quant-sweep.yaml --fake --out results/fake.json --plot results/fake.png`
Expected: prints a markdown table with fp16/q8_0/q4_k_m rows; `results/fake.json` and `results/fake.png` exist.

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: all unit tests PASS, 1 smoke test SKIPPED.

- [ ] **Step 8: Commit**

```bash
git add src/inference_lab/backends/llama_cpp.py scripts/run.py configs/quant-sweep.yaml tests/test_llama_smoke.py
git commit -m "feat: llama.cpp backend + CLI + quant-sweep config (fake path verified)"
```

---

## Post-plan (not tasks — real-run + writeup, after the harness is green)

1. `uv pip install -e '.[llama]'` with Metal: `CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python`.
2. Download the 3 GGUF quants into `./models/` (README documents exact source).
3. `uv run python scripts/run.py --config configs/quant-sweep.yaml` → real numbers.
4. Write `README.md`: methodology + results table + plot + analysis of the speed/memory/quality tradeoff.

## Self-Review

- **Spec coverage:** config ✓(T3) · Backend abstraction ✓(T1,T4,T7) · metrics TTFT/throughput/memory/quality ✓(T2,T5) · runner+persistence ✓(T5) · report table+plot ✓(T6) · llama.cpp Metal ✓(T7) · unit tests deterministic ✓ · smoke skipped w/o model ✓(T7) · default 1.5B / quant labels ✓(T7 config). Quality-guard `output_similarity` is defined (T2) and outputs are captured per variant (T5) for the writeup's fp16-vs-quant comparison.
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type consistency:** `GenerationResult`, `RunConfig`, `RunMetrics`, `RunResult`, `Backend.generate(prompt, max_tokens, seed)` used identically across tasks.
