# Compiler-as-Reward: Operating-Regime Conditions for Agentic RL on Verifiable Coding Tasks

Using compiler/validator feedback as a deterministic, annotation-free reward signal for RL post-training of multi-turn coding agents.

📄 **Paper**: [Compiler-as-Reward: Operating-Regime Conditions for Agentic RL on Verifiable Coding Tasks](paper/draft.pdf) (Junhao Fu, 2026)

## Key Idea

This is an **agentic RL** setting — multi-turn tool use (5 verbs), env-state changes (Sitemuse validator state), sparse terminal reward — closer to [Cursor Composer 2](https://cursor.com/resources/Composer2.pdf), [MiniMax-M1](https://arxiv.org/abs/2506.13585), and [Kimi K2](https://arxiv.org/abs/2507.20534) than to reasoning-RL (DeepSeek-R1 / MiMo-7B math/code single-turn).

Compilers and validation APIs provide **free, deterministic** reward signals — no preference data, no reward model, no LLM judge. We test how far this gets on a real Shopify Horizon template-generation task with a live Sitemuse `themeFilesUpsert` validator, and find:

> **The bottleneck is operating regime, not reward shape.** A single in-context schema example lifts a 4B base model's compile-pass rate from **7.8% → 73.4%** *before any RL training*. In this primed regime, plain GiGPO with binary outcome reward is sufficient. Three process-reward variants — hand-designed step credit (Compiler-OPD, theoretical Goodhart), SFT-on-winners (Best-Trajectory Distillation, **−22.7pp on OOD**), and Cursor-style auxiliary product penalties (v3-shaped, **stagnant at base**) — all fail to beat the minimal binary baseline in this compute-constrained setting, consistent with [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290)'s finding that minimalist reward outperforms complex shaping.

## Main Result

Three eval distributions × three methods (single seed, Qwen3-4B-Instruct-2507):

| Method | val (n=50) | test (n=50) | eval_fixed (n=22, OOD) | weighted |
|--------|-----------:|------------:|-----------------------:|---------:|
| Base (no RL) | 70.0 | 74.0 | 40.9 | 61.6 |
| **v1 — GiGPO + binary outcome** | **78.0** | **76.0** | **50.0** | **68.0** |
| v2-distill — + SFT-on-winners (α=0.1) | 74.0 | 70.0 | 27.3 | 57.1 |
| v3-shaped — + Cursor-style product penalties | 71.9 (s30) | — | — | — |

`eval_fixed` is the OOD modify/add/create task set from prior baseline work; v1 trained only on from-scratch generation prompts.

**v3-shaped (Cursor Composer 2-style auxiliary penalties)** stalled at the base rate over 40 training steps on in-loop validation (n=32 fixed prefix). Full n=122 eval was blocked by AutoDL SSH outage; LoRA archived at step_20 / step_40 for post-hoc evaluation. The negative result confirms Open-Reasoner-Zero's finding that minimalist outcome reward outperforms hand-designed shaping in compute-constrained settings.

Full per-task trajectories in `experiments/logs/v1_horizon/` and `experiments/logs/v2_distill/`.

## What This Repo Contains

```
├── paper/                            # 4-page workshop paper (rewritten draft + PDF)
│   ├── draft.tex
│   └── draft.pdf
├── environments/
│   ├── horizon_env.py                # Local theme inspection (list/describe section)
│   ├── horizon_env_multiturn.py      # Multi-turn HorizonAgentEnv (gym-style step/reset)
│   └── sitemuse_validator.py         # Shopify themeFilesUpsert API client
├── data/
│   ├── prepare.py                    # Synthetic prompt generator (423 train / 50 val)
│   ├── prepare_eval_set.py           # OOD eval_fixed.jsonl (22 modify/add/create tasks)
│   └── prompts/{train,val,test,eval_fixed}.jsonl
├── rewards/                          # Reward functions used during exploration
│   ├── compiler_opd_dpo.py           # DPO-style OPD (theoretical, not run end-to-end)
│   └── compiler_opd.py               # Token-level OPD (legacy, broken in earlier iter)
├── experiments/
│   └── logs/
│       ├── v1_horizon/               # GiGPO + binary outcome (Horizon)
│       │   ├── README.md             # Run config + bug-fix history
│       │   ├── RESULTS.md            # Final eval writeup
│       │   ├── eval_base.json
│       │   ├── eval_step20.json
│       │   └── horizon_train1_v1_*.log
│       └── v2_distill/               # GiGPO + SFT-on-winners (α=0.1)
│           ├── RESULTS.md
│           ├── eval_step60.json
│           └── horizon_train2_distill.log
├── docs/
│   ├── CLAUDE_REVIEW_PATCH.md        # OPD-DPO + multi-turn env design notes
│   ├── baseline-results.md           # Pre-RL baselines (Claude/Qwen zero-shot)
│   └── gpu-server-setup.md           # AutoDL setup notes
└── scripts/
    └── run_verl_grpo.sh              # GRPO launch script (reference)
```

The actual RL training was run via the `verl-agent` fork at
[`johnhaofu/verl-agent`](https://github.com/johnhaofu/verl-agent), branch `horizon-integration`,
which contains:

- `agent_system/environments/env_package/horizon/` — Ray multi-process env wrapper
- `agent_system/environments/prompts/horizon.py` — system prompt with the few-shot priming example
- `examples/gigpo_trainer/run_horizon_lora.sh` — v1 launch script
- `examples/gigpo_trainer/run_horizon_lora_v2_distill.sh` — v2-distill launch script
- `eval_horizon.py` — multi-turn eval script (matches training rollout exactly)

## Method Variants

| ID | Description | Status |
|----|-------------|--------|
| **v1** | GiGPO + binary compile reward + few-shot priming | ✅ run, eval done — current best |
| v2-OPD | + per-step `+0.1` reward on first 2 `fix` calls (Math-Shepherd-style) | ❌ not run; cap-K is hand-tuned, Goodhart risk; sparseness fix obviated by priming |
| v2-distill | + SFT auxiliary loss on tokens of trajectories with reward ≥ 0.5, α=0.1 | ✅ run, eval done — underperforms v1 |
| v3-essa | GiGPO + Error-State Step Anchor (replaces obs-string anchor with canonical validator-error fingerprint) | ✅ run to step 40, in-loop val 70-81% (n=32); full eval pending |
| v3-shaped | + Cursor Composer 2-style product penalties (empty-section, unused-describe, used-describe-bonus, repeat-submit) | ✅ run to step 40; stalled at base rate in-loop, negative result |
| v4 (future) | Either: (a) DAPO dynamic sampling + Clip-Higher (MiMo-7B recipe), (b) phi-4-style Sonnet-regenerated few-shot, or (c) store-context injection à la `TemplateFillAgent` | 🚧 design notes in `docs/v4_design_options.md` |

## Environment

- **Task**: Generate a Shopify Horizon-compatible template JSON satisfying a natural-language design request.
- **Action grammar (5 verbs)**: `list_sections[]`, `describe_section[<name>]`, `describe_block[<name>]`, `fix[<json>]`, `submit[<json>]`. `max_steps=6`.
- **Reward**: Sitemuse `themeFilesUpsert` GraphQL API. Binary pass/fail, deterministic. Free.
- **Datasets**: 423 train + 50 val + 50 held-out test (same distribution, seed=1337) + 22 eval_fixed (OOD).

## Quick Start (eval an existing LoRA)

```bash
# Set up Sitemuse credentials (see environments/sitemuse_validator.py)
export SITEMUSE_TOKEN=...
export SITEMUSE_SHOP_ID=...
export HORIZON_THEME_ID=...

# Eval base model on all three distributions
python eval_horizon.py --checkpoint "" --dataset all --output eval_base.json

# Eval v1 LoRA
python eval_horizon.py \
    --checkpoint /path/to/global_step_20/actor/lora_adapter \
    --dataset all \
    --output eval_v1.json
```

## Model

- **Base**: `Qwen/Qwen3-4B-Instruct-2507` (non-thinking variant; native 262k context; per-spec does NOT emit `<think>` blocks)
- **Adapter**: LoRA rank 32, all attention layers
- **Training**: GiGPO via [verl-agent](https://github.com/johnhaofu/verl-agent), 76 (v1) / 73 (v2-distill) steps before disk-full / collapse respectively

## Caveats

- **Single seed**: all numbers are point estimates; multi-seed runs are the obvious next step.
- **Small batch (64 episodes/step)**: concurrent work from [Ramp Labs Fast Ask](https://ramp.com/labs/fast-ask) uses 2048 with vanilla GRPO and reports no instability; we suspect our v2-distill late collapse and v3-shaped stagnation are both partly a small-batch effect. MiMo-7B ([arXiv:2505.07608](https://arxiv.org/abs/2505.07608)) uses batch=512 on the same `verl` framework and applies DAPO dynamic sampling + Clip-Higher to maintain stability — both are one-line config changes in `verl` and the obvious next intervention.
- **Sitemuse API auth**: token rotated mid-experiment; re-issue if encountering 401.
- **OPD never run end-to-end**: theoretical analysis only, due to Goodhart concerns. The DPO-style alternative (`rewards/compiler_opd_dpo.py`) is implemented but not evaluated.
- **v3-shaped eval incomplete**: AutoDL SSH outage during scheduled evaluation; step_20 and step_40 LoRAs archived to `autodl-fs:horizon_lora_archive/` for post-hoc evaluation on the full n=122 prompts.
- **Few-shot itself is an empty shell**: the in-context example (commit `f4d8543`) that lifted base from 7.8% → 73.4% has empty `settings: {}` and `blocks: {}` — the same hack v3-shaped's penalties were designed to suppress. Phi-4-Reasoning ([arXiv:2504.21318](https://arxiv.org/abs/2504.21318)) suggests using a stronger teacher (Sonnet/o3-mini) to regenerate the few-shot — a 1-day fix that would simultaneously raise the base and remove v3-shaped's underlying motivation.

## Citation

```bibtex
@article{fu2026operatingregime,
  title={Compiler-as-Reward: Operating-Regime Conditions for Agentic RL on Verifiable Coding Tasks},
  author={Fu, Junhao},
  year={2026},
  note={Workshop paper, draft}
}
```

## License

MIT
