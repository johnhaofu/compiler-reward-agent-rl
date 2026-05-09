# Horizon v1 training logs (GiGPO + LoRA + few-shot prompt)

In-flight snapshot of the v1 RL run on AutoDL A800. Training started 2026-05-08 ~13:53 (after 4 launch iterations to fix bugs).

## Files

- `horizon_train1_v1_inflight.log` — full stdout from `verl.trainer.main_ppo`. Includes Hydra config dump + per-step metrics line + sample episode rollouts (verbose).
- `horizon_train1_v1_metrics.log` — filtered to step metrics + val checkpoints + training progress bar. Use this for plotting.

## Config (matches `examples/gigpo_trainer/run_horizon_lora.sh@10702fc`)

- Base model: `Qwen3-4B-Instruct-2507` (non-thinking)
- Algorithm: GiGPO (`mean_norm`), no KL, group_n=8
- LoRA: rank=32, alpha=32, lr=3e-6
- Batch: train=8 prompts × 8 group = 64 episodes / step
- Reward: binary (Sitemuse upsert pass/fail) + −0.1 invalid action penalty
- Env: `HorizonAgentEnv` multi-turn, max_steps=6, history_length=4
- Sampling: temperature=1.0 (rollout), 0.4 (val_kwargs)
- val_freq: every 10 steps on val.jsonl[:32] (deterministic prefix)
- save_freq: every 20 steps (LoRA adapter only)
- Total: 106 epochs (~9.5h ETA)

## Bug fixes that landed BEFORE this log

1. `{% schema %}` literal in prompt template → escaped `{{% schema %}}` (commit `334cbbe`)
2. `<think>` requirement in `horizon_projection` → dropped (Qwen3-Instruct-2507 emits `<tool_call>` instead) (commit `fa75651`)
3. `<think>` instruction in HORIZON_TEMPLATE prompt → removed (commit `b3dd295`)
4. **Few-shot template example** added to HORIZON_TEMPLATE → lifted base success rate **7.8% → 73.4%** (commit `f4d8543`)

The in-flight log is the run that started post-fix-#4. Earlier crashed/buggy logs are NOT included.

## Key data points (in-flight)

| step | val/success_rate | val/text/test_score |
|------|------------------|---------------------|
| 0    | 73.4%            | 64.1%               |
| 10   | 71.9%            | 68.6%               |
| 20   | 78.1%  (+4.7pp)  | 70.1%  (+6.0pp)     |

Step 18-26 train success: peaks at 85.9% (step 26), 5-step rolling 77.7%.

## How to read the metrics log

Each step prints one line starting with `step:N - ...` with these fields:
- `episode/success_rate` — % of 64 episodes that compiled cleanly
- `episode/reward/mean` — average episode reward (with −0.1 invalid penalty)
- `episode/valid_action_ratio` — should be ~0.9+ post-fix (was 0.0 with the bug)
- `episode/length/mean` — average turns per episode
- `actor/grad_norm`, `actor/pg_loss`, `actor/entropy_loss` — RL diagnostics
- `val/text/test_score`, `val/success_rate` — present at val_freq=10 steps only
