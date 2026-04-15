#!/bin/bash
# Evaluate a model on the 22-task agent benchmark
set -e

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

MODEL_PATH=${1:?Usage: run_eval.sh <model_path> <run_name> [run_id]}
RUN_NAME=${2:?Usage: run_eval.sh <model_path> <run_name> [run_id]}
RUN_ID=${3:-1}

# Ensure Horizon is available
git clone --depth 1 https://github.com/Shopify/horizon.git /root/autodl-tmp/horizon 2>/dev/null || true

echo "=== Starting vLLM server: ${MODEL_PATH} ==="
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

nohup vllm serve ${MODEL_PATH} \
  --port 8000 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  > /root/autodl-tmp/vllm_server.log 2>&1 &

echo "Waiting for vLLM to start..."
for i in $(seq 1 60); do
  if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM ready!"
    break
  fi
  sleep 5
done

echo "=== Running eval: ${RUN_NAME} ==="
python experiments/eval_qwen_baseline.py \
  --api-base http://localhost:8000/v1 \
  --horizon-path /root/autodl-tmp/horizon \
  --max-samples 22 --max-turns 50 \
  --output-path experiments/results/${RUN_NAME}.json \
  --run-id ${RUN_ID}

echo "=== Eval Done: ${RUN_NAME} ==="
pkill -f "vllm serve" 2>/dev/null || true
