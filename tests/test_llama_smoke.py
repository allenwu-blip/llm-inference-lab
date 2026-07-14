import os
import pytest

MODEL = os.environ.get("LLAMA_TEST_MODEL")


@pytest.mark.skipif(not MODEL or not os.path.exists(MODEL),
                    reason="set LLAMA_TEST_MODEL to a small GGUF to run")
def test_llama_backend_generates():
    from inference_lab.backends.llama_cpp import LlamaCppBackend
    b = LlamaCppBackend(MODEL, n_ctx=512)
    b.load()
    r = b.generate("Say hi in one word.", max_tokens=8, seed=1)
    assert r.generated_tokens >= 1
    assert r.total_s >= r.ttft_s > 0
    assert isinstance(r.output_text, str) and len(r.output_text) > 0
