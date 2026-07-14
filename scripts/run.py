from __future__ import annotations
import argparse, sys
sys.path.insert(0, "src")
from inference_lab.config import load_experiment
from inference_lab.runner import run_experiment, run_experiment_isolated, save_results
from inference_lab.report import render_table, render_plot
from inference_lab.backends.fake import FakeBackend
from inference_lab.backends.llama_cpp import LlamaCppBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/run.json")
    ap.add_argument("--plot", default="results/plot.png")
    ap.add_argument("--fake", action="store_true", help="use FakeBackend (no model)")
    ap.add_argument("--isolate", action="store_true",
                    help="run each variant in a fresh process (honest per-variant peak RSS)")
    args = ap.parse_args()

    runs = load_experiment(args.config).expand()
    backend_name = "fake" if args.fake else "llama"
    if args.isolate:
        results = run_experiment_isolated(runs, backend_name)
    else:
        factory = (lambda rc: FakeBackend()) if args.fake else (lambda rc: LlamaCppBackend(rc.model_path))
        results = run_experiment(runs, backend_factory=factory)
    save_results(results, args.out)
    render_plot(results, args.plot)
    print(render_table(results))
    print(f"\nsaved: {args.out}  plot: {args.plot}")


if __name__ == "__main__":
    main()
