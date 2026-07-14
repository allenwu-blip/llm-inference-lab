from __future__ import annotations
from dataclasses import dataclass
import yaml


@dataclass
class RunConfig:
    name: str
    model_path: str
    quant: str
    prompts: list[str]
    max_tokens: int
    seed: int


@dataclass
class ExperimentConfig:
    name: str
    prompts: list[str]
    max_tokens: int
    seed: int
    variants: dict[str, str]

    def expand(self) -> list[RunConfig]:
        return [
            RunConfig(
                name=f"{self.name}:{quant}", model_path=path, quant=quant,
                prompts=self.prompts, max_tokens=self.max_tokens, seed=self.seed,
            )
            for quant, path in self.variants.items()
        ]


def load_experiment(path: str) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    exp = ExperimentConfig(
        name=data.get("name", "experiment"),
        prompts=list(data.get("prompts", [])),
        max_tokens=int(data.get("max_tokens", 0)),
        seed=int(data.get("seed", 0)),
        variants=dict(data.get("variants", {})),
    )
    if not exp.prompts:
        raise ValueError("config: prompts must be non-empty")
    if exp.max_tokens <= 0:
        raise ValueError("config: max_tokens must be > 0")
    if not exp.variants:
        raise ValueError("config: variants must be non-empty")
    return exp
