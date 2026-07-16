#!/bin/bash
# Submit with: sbatch scripts/greatlakes.sbatch
# Assumes the one-time setup (uv sync + CUDA wheel + models) is already done in
# $HOME/llm-inference-lab — see docs/running-on-greatlakes.md.
#SBATCH --job-name=llm-inference-lab
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source $HOME/.local/bin/env
cd $HOME/llm-inference-lab
nvidia-smi -L
uv run python scripts/run.py --config configs/quant-sweep-0.5b.yaml --isolate
