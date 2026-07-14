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


def run_single(rc: RunConfig, backend_name: str = "llama") -> dict:
    """Run one variant and return a JSON-able payload.

    Peak RSS is measured with this process's own getrusage, so when each
    variant is run in a fresh process (see run_experiment_isolated) the
    memory number reflects that variant alone — not the whole sweep.
    """
    if backend_name == "fake":
        from .backends.fake import FakeBackend
        backend: Backend = FakeBackend()
    else:
        from .backends.llama_cpp import LlamaCppBackend
        backend = LlamaCppBackend(rc.model_path)
    backend.load()
    gens = [backend.generate(p, rc.max_tokens, rc.seed) for p in rc.prompts]
    return {
        "variant": rc.quant,
        "metrics": asdict(aggregate(gens)),
        "peak_rss_mb": _peak_rss_mb(),
        "model_size_mb": _model_size_mb(rc.model_path),
        "outputs": [g.output_text for g in gens],
    }


def run_experiment_isolated(runs: list[RunConfig], backend_name: str = "llama") -> list[RunResult]:
    """Run each variant in a fresh process so peak RSS is honest per variant."""
    import multiprocessing as mp
    from .metrics import RunMetrics
    ctx = mp.get_context("spawn")
    payloads: list[dict] = []
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        payloads = pool.starmap(run_single, [(rc, backend_name) for rc in runs])
    return [
        RunResult(
            variant=p["variant"], metrics=RunMetrics(**p["metrics"]),
            peak_rss_mb=p["peak_rss_mb"], model_size_mb=p["model_size_mb"],
            outputs=p["outputs"],
        )
        for p in payloads
    ]


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
