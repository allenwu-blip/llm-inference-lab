from __future__ import annotations
from .base import GenerationResult


class FakeBackend:
    def __init__(self, ttft_s: float = 0.05, per_token_s: float = 0.01, name: str = "fake"):
        self.ttft_s = ttft_s
        self.per_token_s = per_token_s
        self.name = name

    def load(self) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int, seed: int) -> GenerationResult:
        total_s = self.ttft_s + self.per_token_s * max_tokens
        text = f"[{seed}] " + " ".join(prompt.split()[:3]) * max(1, max_tokens // 4)
        return GenerationResult(
            prompt=prompt, output_text=text,
            prompt_tokens=len(prompt.split()), generated_tokens=max_tokens,
            ttft_s=self.ttft_s, total_s=total_s,
        )
