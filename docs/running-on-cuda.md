# Reproducing the sweep on an NVIDIA GPU (free, via Google Colab)

The harness is engine-portable — the same code runs on Apple-Silicon (Metal) and NVIDIA
(CUDA), so the only thing that changes is the backend build. This reproduces the
quantization sweep on a CUDA GPU for a **cross-hardware comparison**, free, on Colab's T4.

## Steps

1. Open a new notebook at <https://colab.research.google.com>.
2. **Runtime → Change runtime type → T4 GPU** (do this first, or it runs on CPU).
3. Paste this into a cell and run it:

```bash
!nvidia-smi -L
!git clone -q https://github.com/allenwu-blip/llm-inference-lab.git
%cd llm-inference-lab
!pip install -q -e .
!CMAKE_ARGS="-DGGML_CUDA=on" pip install -q llama-cpp-python
!mkdir -p models && for q in fp16 q8_0 q4_k_m; do wget -q -O models/qwen2.5-0.5b-instruct-$q.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-$q.gguf; done
!python scripts/run.py --config configs/quant-sweep-0.5b.yaml --isolate
```

The CUDA build of `llama-cpp-python` takes a few minutes. The final output is the results
table — the CUDA counterpart of the Apple-M3/Metal numbers in the main README.

> T4 has 16 GB of VRAM (vs the M3's 8 GB shared), so it can also run larger models —
> point the config at a 1.5B / 3B GGUF to widen the fp16↔q4 gap.
