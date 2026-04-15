#!/bin/bash
# Full experiment pipeline: B1→B2→B3→B4→M1→M2→M3, each followed by eval
set -e

export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UNSLOTH_USE_MODELSCOPE=1

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

echo "========================================"
echo "  Step 1: B1 — SFT"
echo "========================================"
bash scripts/run_sft.sh
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-sft qwen3.5_b1_sft 1

echo "========================================"
echo "  Step 2: B2 — GRPO outcome reward"
echo "========================================"
bash scripts/run_grpo.sh outcome
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-grpo-outcome qwen3.5_b2_grpo 1

echo "========================================"
echo "  Step 3: B3 — GRPO multi-signal reward"
echo "========================================"
bash scripts/run_grpo.sh multi_signal
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-grpo-multi_signal qwen3.5_b3_grpo_multi 1

echo "========================================"
echo "  Step 4: B4 — GSPO"
echo "========================================"
bash scripts/run_b4.sh
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-gspo qwen3.5_b4_gspo 1

echo "========================================"
echo "  Step 5: M1 — GSPO + Compiler-OPD"
echo "========================================"
bash scripts/run_m1.sh
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-grpo-opd qwen3.5_m1_opd 1

echo "========================================"
echo "  Step 6: M2 — GSPO + Error-Branch"
echo "========================================"
bash scripts/run_m2.sh
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-error-branch qwen3.5_m2_branch 1

echo "========================================"
echo "  Step 7: M3 — Full Method"
echo "========================================"
bash scripts/run_m3.sh
bash scripts/run_eval.sh /root/autodl-tmp/models/qwen3.5-full-method qwen3.5_m3_full 1

echo "========================================"
echo "  All experiments complete!"
echo "========================================"
ls -la experiments/results/qwen3.5_*.json
