#!/bin/bash
# verl-agent GRPO training with multi-turn agent rollout (SGLang + BaseTool).
#
# Architecture:
#   - 3x RTX 4080 SUPER (32GB each, 96GB total)
#   - SGLang for rollout (handles multi-turn tool calling)
#   - HorizonTool (BaseTool) for each of 9 Horizon agent tools
#   - Reward: step rewards (validate=+0.3, done=+0.3, error=-0.05) +
#             final reward (resolved=1.0, first_try_valid=0.5, else=0.0)
#
# Usage:
#   bash scripts/run_verl_grpo.sh [outcome|multi_signal] [grpo|gspo|gigpo]

set -ex

# Use verl-agent dedicated environment
export VENV=/root/autodl-tmp/.venv-verl
export PATH=$VENV/bin:$PATH
export TMPDIR=/root/autodl-tmp/tmp
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export PYTHONPATH=/root/autodl-tmp/compiler-reward-agent-rl:/root/autodl-tmp/verl-agent:$PYTHONPATH
export HORIZON_PATH=/root/autodl-tmp/horizon

ALGO=${1:-grpo}                  # grpo | gspo | gigpo
GROUP_SIZE=${2:-4}               # generations per prompt for GRPO
N_GPUS=${N_GPUS:-3}              # can override via env var
TRAIN_SIZE=${TRAIN_SIZE:-48}     # must be divisible by N_GPUS
VAL_SIZE=${VAL_SIZE:-24}         # must be divisible by N_GPUS

cd /root/autodl-tmp/compiler-reward-agent-rl
git pull

# Ensure Horizon theme exists
git clone --depth 1 https://github.com/Shopify/horizon.git $HORIZON_PATH 2>/dev/null || true

# Sync our integration files into verl-agent's tree
cp -r agent_system/environments/env_package/horizon \
      /root/autodl-tmp/verl-agent/agent_system/environments/env_package/
cp verl_tools/horizon_tool.py /root/autodl-tmp/verl-agent/verl/tools/
cp verl_tools/horizon_tool_config.yaml /root/autodl-tmp/verl-agent/

# Patch env_manager.py to register horizon (idempotent)
$VENV/bin/python -c "
path = '/root/autodl-tmp/verl-agent/agent_system/environments/env_manager.py'
with open(path) as f:
    content = f.read()

needle = 'if \"horizon\" in config.env.env_name.lower():'
if needle not in content:
    insert = '''
    if \"horizon\" in config.env.env_name.lower():
        import sys; sys.path.insert(0, \"/root/autodl-tmp/compiler-reward-agent-rl\")
        from agent_system.environments.env_package.horizon import build_horizon_envs, horizon_projection
        _envs = build_horizon_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_horizon_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)
        projection_f = partial(horizon_projection)
        envs = EnvironmentManagerBase(_envs, projection_f, config)
        val_envs = EnvironmentManagerBase(_val_envs, projection_f, config)
        return envs, val_envs

'''
    content = content.replace(
        '    if \"search\" in config.env.env_name.lower():',
        insert + '    if \"search\" in config.env.env_name.lower():',
        1
    )
    with open(path, 'w') as f:
        f.write(content)
    print('env_manager patched')
else:
    print('env_manager already has horizon')
"

# Prepare RL data (prompts only, in messages format)
$VENV/bin/python -c "
import json, os, pandas as pd
from pathlib import Path

SYS = ('You are an expert Shopify theme developer working with the Horizon theme. '
       'Use the provided tools to research, generate, and validate template files. '
       'Always validate before calling done.')

def load(path, n=None):
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n: break
            it = json.loads(line)
            items.append({
                'prompt': [
                    {'role': 'system', 'content': SYS},
                    {'role': 'user', 'content': it['prompt'][-1]['content']},
                ],
                'ground_truth': it.get('ground_truth', ''),
                'data_source': 'horizon',
            })
    return items

train = load('data/prompts/train.jsonl', $TRAIN_SIZE)
val = load('data/prompts/eval_fixed.jsonl', $VAL_SIZE)

out = Path(os.environ['HOME']) / 'data/verl-agent/text'
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(train).to_parquet(out / 'train.parquet')
pd.DataFrame(val).to_parquet(out / 'test.parquet')
print(f'Train: {len(train)}, Val: {len(val)}')
"

# Kill any stale processes
pgrep -f main_ppo 2>/dev/null | xargs -r kill -9 2>/dev/null || true
pgrep -f sglang 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 5

EXPNAME="${ALGO}_qwen3_horizon_multiturn"

echo "=== verl-agent ${ALGO^^} + SGLang multi-turn (3 GPU) ==="
$VENV/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ALGO} \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_SIZE \
    data.val_batch_size=$VAL_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_turns=50 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/root/autodl-tmp/verl-agent/horizon_tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.format=chatml \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=horizon \
    env.seed=42 \
    env.max_steps=50 \
    env.rollout.n=$GROUP_SIZE \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='compiler_reward_agent' \
    trainer.experiment_name="$EXPNAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.val_before_train=True
