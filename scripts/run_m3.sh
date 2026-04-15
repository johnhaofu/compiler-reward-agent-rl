#!/bin/bash
# M3: Full Method (GSPO + OPD + Error-Branch)
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

echo "=== M3: Full Method ==="
python training/full_method_train.py \
  --model-name /root/autodl-tmp/models/qwen3.5-sft \
  --horizon-path /root/autodl-tmp/horizon \
  --output-dir /root/autodl-tmp/models/qwen3.5-full-method \
  --epochs 1 --lr 5e-6 --batch-size 2 --grad-accum 4 \
  --num-generations 8 --branch-generations 2 \
  --temperature 0.8 --branch-temperature 0.5 \
  --opd-weight 0.3 --recovery-bonus 0.2 --gspo-margin 0.5

echo "=== M3 Done ==="
