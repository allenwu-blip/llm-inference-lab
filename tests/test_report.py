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
