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
