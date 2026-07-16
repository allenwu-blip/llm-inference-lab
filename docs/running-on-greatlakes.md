# Running the sweep on UMich Great Lakes (NVIDIA V100, via Slurm)

Great Lakes is the University of Michigan's HPC cluster. This runs the same harness on a
real datacenter GPU (V100) through Slurm — a third hardware point alongside Apple M3
(Metal) and the Colab T4.

**Prereqs:** a Great Lakes login and a Slurm account you can charge (College of
Engineering students can be added to the shared `engin1` account by ARC Support). Off
campus, connect to UMVPN first.

## 1. SSH in

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu   # then approve the Duo push
```

## 2. One-time setup (on the LOGIN node — it has internet)

```bash
# uv manages Python + deps, so no Lmod modules to wrangle
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/allenwu-blip/llm-inference-lab.git
cd llm-inference-lab
uv sync
uv pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --prefer-binary

# 0.5B quants (same as the M3 / T4 runs, for an apples-to-apples third data point)
mkdir -p models
for q in fp16 q8_0 q4_k_m; do
  wget -q -O models/qwen2.5-0.5b-instruct-$q.gguf \
    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-$q.gguf
done
```

## 3. Grab a GPU and run

```bash
salloc --account=engin1 --partition=gpu --gpus=1 --cpus-per-task=4 --mem=16G --time=00:30:00
# once Slurm drops you onto a GPU node:
cd llm-inference-lab
nvidia-smi -L                                          # confirm the V100
uv run python scripts/run.py --config configs/quant-sweep-0.5b.yaml --isolate
```

The final table is the V100 counterpart of the M3-Metal and T4-CUDA numbers.

## Batch version (reproducible, hands-off)

After the one-time setup, submit the job instead of sitting in an interactive shell:

```bash
sbatch scripts/greatlakes.sbatch     # results land in slurm-<jobid>.out
```

## Notes / fallbacks

- If the prebuilt CUDA wheel won't import on the GPU node, build from source for the V100:
  `module load cuda && CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70" uv pip install --no-binary=llama-cpp-python llama-cpp-python`  (V100 = compute capability 7.0).
- `--partition=gpu` = V100. A40s are on `--partition=spgpu`; ask ARC which partition holds the A100s.
- If `module load python` is ever needed, `module spider python` lists the exact versions.
- Slurm reference: <https://arc.umich.edu/greatlakes/slurm-user-guide/>
