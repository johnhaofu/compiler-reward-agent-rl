#!/bin/bash
# M2: GSPO + Error-Branch
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

echo "=== M2: GSPO + Error-Branch ==="
python training/error_branch_train.py \
  --model-name /root/autodl-tmp/models/qwen3.5-sft \
  --horizon-path /root/autodl-tmp/horizon \
  --output-dir /root/autodl-tmp/models/qwen3.5-error-branch \
  --epochs 1 --lr 5e-6 --batch-size 2 --grad-accum 4 \
  --num-generations 4 --branch-generations 2 \
  --temperature 0.7 --branch-temperature 0.5 --recovery-bonus 0.2

echo "=== M2 Done ==="
