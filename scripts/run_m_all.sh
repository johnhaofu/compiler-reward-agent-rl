#!/bin/bash
# Full M-series experiments: M1 (OPD) → eval
# Requires B1 (SFT) to be completed first
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

echo "========================================"
echo "  M1: GRPO + Compiler-OPD"
echo "========================================"
bash scripts/run_m1.sh

echo "========================================"
echo "  Eval M1"
echo "========================================"
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-grpo-opd qwen3.5_m1_opd 1

echo "========================================"
echo "  All M experiments complete!"
echo "========================================"
ls -la experiments/results/qwen3.5_*.json
