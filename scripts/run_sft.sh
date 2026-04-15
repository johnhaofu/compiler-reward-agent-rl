#!/bin/bash
# B1: SFT Training on expert trajectories
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

echo "=== B1: SFT Training ==="
python training/sft_train.py \
  --model-name /root/autodl-tmp/models/Qwen3.5-4B \
  --data-path data/sft/train.jsonl \
  --output-dir /root/autodl-tmp/models/qwen3.5-sft \
  --epochs 3 --lr 2e-4 --batch-size 2 --grad-accum 4 \
  --max-seq-len 8192 --lora-r 16

echo "=== SFT Done ==="
