#!/bin/bash
# Full experiment pipeline: B1 (SFT) → eval → B2 (GRPO) → eval → B3 (GRPO multi) → eval
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

echo "========================================"
echo "  Step 1: SFT Training (B1)"
echo "========================================"
bash scripts/run_sft.sh

echo "========================================"
echo "  Step 2: Eval B1 (SFT)"
echo "========================================"
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-sft qwen3.5_sft_b1 1

echo "========================================"
echo "  Step 3: GRPO outcome reward (B2)"
echo "========================================"
bash scripts/run_grpo.sh outcome

echo "========================================"
echo "  Step 4: Eval B2 (GRPO outcome)"
echo "========================================"
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-grpo-outcome qwen3.5_grpo_b2 1

echo "========================================"
echo "  Step 5: GRPO multi-signal reward (B3)"
echo "========================================"
bash scripts/run_grpo.sh multi_signal

echo "========================================"
echo "  Step 6: Eval B3 (GRPO multi-signal)"
echo "========================================"
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-grpo-multi_signal qwen3.5_grpo_b3 1

echo "========================================"
echo "  All experiments complete!"
echo "  Results in experiments/results/"
echo "========================================"
ls -la experiments/results/qwen3.5_*.json
