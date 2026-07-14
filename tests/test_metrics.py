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
