from __future__ import annotations
import time
from .base import GenerationResult


class LlamaCppBackend:
    def __init__(self, model_path: str, n_ctx: int = 2048, name: str = "llama.cpp"):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.name = name
        self._llm = None

    def load(self) -> None:
        from llama_cpp import Llama  # imported lazily so unit tests don't need it
        self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx,
                          n_gpu_layers=-1, verbose=False)

    def generate(self, prompt: str, max_tokens: int, seed: int) -> GenerationResult:
        assert self._llm is not None, "call load() first"
        start = time.perf_counter()
        first_t: float | None = None
        pieces: list[str] = []
        n = 0
        stream = self._llm(prompt, max_tokens=max_tokens, seed=seed, stream=True)
        for chunk in stream:
            if first_t is None:
                first_t = time.perf_counter()
            pieces.append(chunk["choices"][0]["text"])
            n += 1
        total_s = time.perf_counter() - start
        ttft_s = (first_t - start) if first_t is not None else total_s
        prompt_tokens = len(self._llm.tokenize(prompt.encode()))
        return GenerationResult(
            prompt=prompt, output_text="".join(pieces),
            prompt_tokens=prompt_tokens, generated_tokens=n,
            ttft_s=ttft_s, total_s=total_s,
        )
