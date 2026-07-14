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
