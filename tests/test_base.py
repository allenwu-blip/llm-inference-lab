from inference_lab.backends.base import GenerationResult


def test_generation_result_holds_fields():
    r = GenerationResult(
        prompt="hi", output_text="hello there",
        prompt_tokens=1, generated_tokens=2, ttft_s=0.1, total_s=0.3,
    )
    assert r.generated_tokens == 2
    assert r.total_s == 0.3
