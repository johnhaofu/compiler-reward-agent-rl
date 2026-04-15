#!/bin/bash
# M1: GRPO + Compiler-OPD
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

echo "=== M1: GRPO + Compiler-OPD ==="
python training/grpo_opd_train.py \
  --model-name /root/autodl-tmp/models/qwen3.5-sft \
  --horizon-path /root/autodl-tmp/horizon \
  --output-dir /root/autodl-tmp/models/qwen3.5-grpo-opd \
  --epochs 1 --lr 5e-6 --batch-size 2 --grad-accum 4 \
  --num-generations 4 --temperature 0.7 \
  --opd-weight 0.3

echo "=== M1 Done ==="
