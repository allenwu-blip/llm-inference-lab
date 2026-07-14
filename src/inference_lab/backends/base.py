from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationResult:
    prompt: str
    output_text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float   # seconds to first generated token
    total_s: float  # seconds for the whole generation call


class Backend(Protocol):
    name: str
    def load(self) -> None: ...
    def generate(self, prompt: str, max_tokens: int, seed: int) -> GenerationResult: ...
