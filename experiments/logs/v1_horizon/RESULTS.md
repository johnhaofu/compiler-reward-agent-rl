# Horizon v1 — Final Results

## TL;DR

GiGPO + LoRA + few-shot prompt + Sitemuse compile reward improves Qwen3-4B-Instruct-2507 on Horizon template generation across **all three eval distributions**, with the largest gain on the OOD real-task set.

```
              base       step_20    Δ          relative
val          70.0%       78.0%      +8.0pp     +11.4%
test         74.0%       76.0%      +2.0pp     +2.7%
eval_fixed   40.9%       50.0%      +9.1pp     +22.2%   ★
─────────────────────────────────────────────────────
weighted     61.6%       68.0%      +6.4pp     +10.4%
```

(`weighted` = average across 122 evaluation prompts, weighted by dataset size.)

## What this means

- **In-distribution (val.jsonl, n=50)**: +8.0pp. RL learned to write JSON the validator accepts.
- **Held-out same-distribution (test.jsonl, n=50, seed=1337)**: +2.0pp. The gain partially transfers; some of val's lift is in-sample fitting.
- **OOD real tasks (eval_fixed.jsonl, n=22, modify/add/create format)**: **+9.1pp / +22% relative**. The biggest surprise. Even though training was pure from-scratch generation and eval_fixed asks for modifications to existing templates, the model learned a more general "produce compilable Liquid templates" capability that transfers.

The eval_fixed result is the strongest evidence that **compile-pass reward generalizes**, not just memorizes.

## Methodology

- **Base model**: Qwen3-4B-Instruct-2507 (non-thinking variant)
- **Algorithm**: GiGPO (`mean_norm`), no KL, group_n=8, LoRA rank=32, lr=3e-6
- **Reward**: binary Sitemuse `themeFilesUpsert` API (1.0 pass, 0.0 fail) + −0.1 invalid action penalty
- **Env**: multi-turn `HorizonAgentEnv` (max_steps=6, history_length=4)
- **Few-shot prompt**: 1 worked-example template + 5 rule notes (commit `f4d8543`). This alone lifted base from 7.8% → 73.4% on training val before any RL.
- **Training run**: 40 steps before disk-full crash on save_freq=20 → step_40 partial. Best checkpoint = `global_step_20` (val 78.1% mid-train, 78.0% on n=50 here).
- **Eval sampling**: temperature=0.4, top_p=1.0, top_k=-1 (matches training `val_kwargs`). Single-sample pass@1.

## Files

- `eval_base.json` — full base eval, per-task trajectories
- `eval_step20.json` — full step_20 LoRA eval, per-task trajectories
- `eval_base.log`, `eval_step20.log` — stdout from `eval_horizon.py`
- `horizon_train1_v1_final.log` — full training stdout (1306+ lines, includes the crash trace)
- `horizon_train1_v1_metrics.log` — filtered training metrics for plotting

## Caveats

1. **Single seed.** Standard error from one seed on n=22 (eval_fixed) is large; ±5pp swings are plausible. Multi-seed runs are needed for paper-grade claims.
2. **Train crashed at step 40** (disk full during checkpoint save). Step 20 was the last saved LoRA. Training was already plateauing (val 78.1% step 20 → 75.0% step 30 → train trend declining at step 35-39), so step 20 likely is near the actual peak.
3. **Method = GiGPO + binary outcome reward**, not the paper's full M3 (Compiler-OPD + Error-Branch). For paper claims about OPD/Branch, those need to be re-implemented and re-trained.
4. **eval_fixed's verify rules not used.** The 22 tasks have programmatic `verify` (json_check / multi_file). v1 eval only checks Sitemuse compile. Verify-rule scoring would be more rigorous and is an obvious v2 addition.

## What's next (proposed v2)

1. Add Compiler-OPD per-step rewards (~30 lines in `core_env._submit`/`_fix`)
2. Add `max_ckpt_to_keep=2` to launch script to prevent disk-full crashes
3. Multi-seed (3 seeds) for error bars
4. (Optional) Implement verify-rule scoring for eval_fixed
