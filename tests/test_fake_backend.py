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
