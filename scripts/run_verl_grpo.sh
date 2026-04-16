#!/bin/bash
# GRPO training via verl-agent framework
# Multi-turn agent RL with vLLM generation
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export PATH=/root/autodl-tmp/.venv/bin:$PATH

ALGO=${1:-grpo}  # grpo, gspo, gigpo
GROUP_SIZE=${2:-4}

train_data_size=16
val_data_size=22

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

# Prepare data (prompts only, no completions)
python3 -c "
import json, pandas as pd
from pathlib import Path

# Training prompts
train = []
with open('data/prompts/train.jsonl') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        train.append({
            'question': item['prompt'][-1]['content'],
            'ground_truth': item.get('ground_truth', ''),
            'data_source': 'horizon',
        })
        if i >= ${train_data_size} - 1:
            break

# Eval prompts
val = []
with open('data/prompts/eval_fixed.jsonl') as f:
    for line in f:
        item = json.loads(line)
        val.append({
            'question': item['prompt'][-1]['content'],
            'ground_truth': '',
            'data_source': 'horizon',
        })

Path('\$HOME/data/verl-agent/text').mkdir(parents=True, exist_ok=True)
pd.DataFrame(train).to_parquet('\$HOME/data/verl-agent/text/train.parquet')
pd.DataFrame(val).to_parquet('\$HOME/data/verl-agent/text/test.parquet')
print(f'Train: {len(train)}, Val: {len(val)}')
"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ALGO} \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen3.5-4B \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=horizon \
    env.seed=42 \
    env.max_steps=50 \
    env.rollout.n=$GROUP_SIZE \
    env.horizon_path=/root/autodl-tmp/horizon \
    env.schemas_dir=data/schemas \
    env.components_path=data/horizon_components.json \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='compiler_reward_agent' \
    trainer.experiment_name="${ALGO}_qwen3.5_4b" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=50 \
    trainer.val_before_train=True "$@"
