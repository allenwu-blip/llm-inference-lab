import json
from inference_lab.config import RunConfig
from inference_lab.backends.fake import FakeBackend
from inference_lab.runner import run_experiment, save_results, RunResult


def _run(quant):
    return RunConfig(name=f"t:{quant}", model_path="none", quant=quant,
                     prompts=["a b c", "d e f"], max_tokens=8, seed=1)


def test_run_experiment_aggregates_per_variant():
    runs = [_run("fp16"), _run("q4_k_m")]
    results = run_experiment(runs, backend_factory=lambda rc: FakeBackend())
    assert [r.variant for r in results] == ["fp16", "q4_k_m"]
    assert results[0].metrics.n == 2               # two prompts
    assert results[0].peak_rss_mb > 0
    assert len(results[0].outputs) == 2


def test_save_results_writes_json(tmp_path):
    runs = [_run("fp16")]
    results = run_experiment(runs, backend_factory=lambda rc: FakeBackend())
    out = tmp_path / "r.json"
    save_results(results, str(out))
    data = json.loads(out.read_text())
    assert data[0]["variant"] == "fp16"
    assert "decode_tps_mean" in data[0]["metrics"]
