# GPU Server Setup Guide

AutoDL RTX 4080 32GB server setup for running experiments.

## Prerequisites

- AutoDL instance with GPU (RTX 4080 32GB or better)
- System disk: 30GB (keep clean, don't install large packages here)
- Data disk: 100GB+ (models, cache, experiments)

## 1. Install Dependencies (use uv, not pip)

```bash
# Always redirect cache to data disk (system disk is only 30GB)
export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
mkdir -p $TMPDIR $UV_CACHE_DIR

# Install uv
pip install uv

# Install vLLM + PyTorch (auto-detects CUDA version)
uv pip install vllm --torch-backend=auto --index-url https://mirrors.aliyun.com/pypi/simple/

# Install experiment dependencies
uv pip install requests jsonschema pyyaml wandb matplotlib pandas openai --index-url https://mirrors.aliyun.com/pypi/simple/
```

## 2. Download Model

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
```

## 3. Clone Repos

```bash
cd /root/autodl-tmp

# Experiment repo
git clone git@github.com:johnhaofu/compiler-reward-agent-rl.git

# Horizon theme
git clone --depth 1 https://github.com/Shopify/horizon.git /root/autodl-tmp/horizon
```

## 4. Start vLLM Server

```bash
vllm serve /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
  --dtype half --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Verify:
```bash
curl http://localhost:8000/v1/models
```

## 5. Run Qwen Baseline

```bash
cd /root/autodl-tmp/compiler-reward-agent-rl
python experiments/eval_qwen_baseline.py \
  --api-base http://localhost:8000/v1 \
  --horizon-path /root/autodl-tmp/horizon \
  --max-samples 2
```

## Troubleshooting

### System disk full
```bash
rm -rf /root/.cache/pip /tmp/pip-* /root/.cache/uv
pip cache purge
# Always use: export TMPDIR=/root/autodl-tmp/tmp
```

### flash_attn issues
vLLM 0.19+ requires flash_attn. If `uv pip install vllm --torch-backend=auto` works, it handles this automatically. If not, use `TMPDIR=/root/autodl-tmp/tmp pip install flash-attn --no-build-isolation`.

### Model download
Use modelscope (China mirror), not huggingface:
```bash
modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
```
