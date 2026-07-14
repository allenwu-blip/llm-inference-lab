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
