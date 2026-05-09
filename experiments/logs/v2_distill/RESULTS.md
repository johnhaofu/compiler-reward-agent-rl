# Horizon v2-distill — Results

## TL;DR

Best-Trajectory Distillation (SFT loss on winning trajectories, mixed with GiGPO PG loss, α=0.1) **lost to v1 binary outcome baseline** across all three eval distributions, with eval_fixed (OOD) showing the largest drop.

```
              base    v1_step_20    v2_step_60     Δ vs v1
val          70.0%    78.0%         74.0%          -4.0pp
test         74.0%    76.0%         70.0%          -6.0pp
eval_fixed   40.9%    50.0%         27.3%          -22.7pp ↓↓
```

## Why distill failed

Theory: SFT on winning trajectories (Tülu 3 / OLMo 3 "on-policy SFT" pattern) avoids OPD's hand-designed step-reward Goodhart trap by only crediting trajectories that already passed.

In practice on this 4B model + 423-prompt training set:
1. **Early gain (steps 1-10)**: in-loop val 73.4 → 79.7. Distill pulls policy toward known-good outputs faster than pure RL.
2. **Mid-stage degradation (steps 30-40)**: val drops to 67-72%. Distill memorizes training-distribution patterns at the cost of generalization.
3. **Apparent recovery (steps 50-60)**: in-loop val rebounds to 82.8 — but this is **memorization on the fixed val.jsonl prefix [0:32]**, not real improvement.
4. **Real eval (n=50 + eval_fixed)**: catastrophic drop on OOD eval_fixed (-22.7pp), confirming the rebound was distribution-specific overfit.
5. **Late collapse (steps 70+)**: train-success crashed to 14% — distill loss had pulled policy into degenerate mode.

## What this means

**Best-Trajectory Distillation is not free of Goodhart's Law.** The hazard is different from OPD's (no reward hacking via redundant validate calls), but the loss still creates a memorization gradient that competes with RL's generalization gradient. With α=0.1 and 60+ steps, memorization wins.

This experiment **does NOT invalidate** the broader idea — possible mitigations:
- α anneal: 0.1 → 0 over training (let RL take over after distill warmup)
- Lower α (0.05) for a smaller pull
- Stop distill loss once train_success > some threshold

But **as run, v2-distill is a paper-grade negative result** that v1 (GiGPO + binary outcome) is hard to beat in this regime.

## Methodology

Same as v1 except:
- Added `actor_rollout_ref.actor.distill_alpha=0.1`
- Added SFT loss on tokens of winning trajectories (`reward >= 0.5`) in `dp_actor.py`
- Added `is_winner` mask to `select_keys` so it propagates from trainer to actor
- All other hyperparameters identical to v1

Training crashed at step 73 (after step 60 LoRA save) with train_success collapsing to 14%. Best LoRA = step_60.

## Files

- `eval_step60.json` — full per-task eval results for v2_step_60 LoRA
- `eval_step60.log` — eval stdout
- `horizon_train2_distill.log` — full training stdout (~2MB; includes [DISTILL DEBUG] traces)
- `RESULTS.md` — this writeup

## Caveats / lessons learned

1. **In-loop val on fixed prefix lies**: val.jsonl[:32] is a tiny window from same distribution as training; high in-loop val + low real-eval-on-different-data is the warning sign of memorization. Future eval should use larger n + a held-out distribution.
2. **API outage during eval**: Sitemuse API went down for ~30min mid-eval, polluting two trial runs. Fixed by retrying after recovery.
3. **Watchdog archive bug**: my checkpoint watchdog used `basename(dirname(dirname(step_dir)))` which extracted project name not exp name → multiple experiments wrote to same archive dir. Hash check (md5) caught this. Fixed mentally for v3 (use experiment_name explicitly).

## Conclusion

v1 (GiGPO + binary outcome reward + few-shot prompt) remains the best Horizon baseline.

For paper: v2-distill becomes an **honest ablation** showing that even principled non-heuristic process signals (SFT-on-winners) don't naively beat pure outcome-reward RL on this task.
