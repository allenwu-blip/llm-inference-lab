import pytest
from inference_lab.config import ExperimentConfig, load_experiment


def _write(tmp_path, text):
    p = tmp_path / "exp.yaml"
    p.write_text(text)
    return str(p)


def test_load_and_expand(tmp_path):
    path = _write(tmp_path, """
name: sweep
max_tokens: 64
seed: 7
prompts: ["Explain gravity.", "Write a haiku."]
variants:
  fp16: models/m-fp16.gguf
  q4_k_m: models/m-q4.gguf
""")
    exp = load_experiment(path)
    runs = exp.expand()
    assert exp.name == "sweep"
    assert len(runs) == 2
    assert {r.quant for r in runs} == {"fp16", "q4_k_m"}
    assert runs[0].max_tokens == 64 and runs[0].seed == 7
    assert runs[0].prompts == ["Explain gravity.", "Write a haiku."]


def test_invalid_config_raises(tmp_path):
    path = _write(tmp_path, "name: bad\nmax_tokens: 0\nprompts: []\nvariants: {}\nseed: 1\n")
    with pytest.raises(ValueError):
        load_experiment(path)
