#!/bin/bash
# B2/B3: GRPO Training with compiler reward
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

REWARD_TYPE=${1:-outcome}  # outcome (B2) or multi_signal (B3)
OUTPUT_DIR=/root/autodl-tmp/models/qwen3.5-grpo-${REWARD_TYPE}

echo "=== GRPO Training (reward=${REWARD_TYPE}) ==="
python training/grpo_train.py \
  --model-name /root/autodl-tmp/models/qwen3.5-sft \
  --data-path data/prompts/train.jsonl \
  --horizon-path /root/autodl-tmp/horizon \
  --reward-type ${REWARD_TYPE} \
  --output-dir ${OUTPUT_DIR} \
  --epochs 1 --lr 5e-6 --batch-size 2 --grad-accum 4 \
  --num-generations 4 --temperature 0.7 \
  --max-prompt-len 1024 --max-completion-len 4096

echo "=== GRPO Done (${REWARD_TYPE}) ==="
