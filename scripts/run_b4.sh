#!/bin/bash
# B4: GSPO Training
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

echo "=== B4: GSPO ==="
python training/gspo_train.py \
  --model-name /root/autodl-tmp/models/qwen3.5-sft \
  --horizon-path /root/autodl-tmp/horizon \
  --output-dir /root/autodl-tmp/models/qwen3.5-gspo \
  --epochs 1 --lr 5e-6 --batch-size 2 --grad-accum 4 \
  --num-generations 8 --temperature 0.8 --gspo-margin 0.5

echo "=== B4 Done ==="
