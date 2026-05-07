# Compiler-as-Reward: Agent RL for Coding Agents

Using compiler/linter feedback as process reward signals for training coding agents via reinforcement learning.

📄 **Paper**: [Compiler-as-Reward: Process Feedback for Coding Agent RL Training](paper/draft.pdf) (Junhao Fu, 2025)

## Key Idea

Compilers provide **free, deterministic, line-level** feedback — a natural reward signal that has been overlooked in Agent RL research. We systematically study how to leverage this signal for training coding agents on real-world Shopify Liquid template generation.

## Contributions

1. **Compiler-Guided OPD**: Using compiler error messages as hints for On-Policy Distillation, providing token-level directional training signals without an LLM judge
2. **Error-Triggered Branching**: Using compilation failure as a branching signal for adaptive exploration (zero computational overhead vs entropy-based methods)
3. **First systematic Agent RL evaluation on real-world coding tasks** (Shopify Liquid, not toy benchmarks)

## Environment

- **Task**: Generate Shopify page templates (JSON + Liquid) based on the [Horizon](https://github.com/Shopify/horizon) theme
- **Input**: Natural language page description
- **Output**: Valid Horizon-compatible template
- **Validation**: `shopify theme check` + JSON schema + section/block reference check

## Experiments

| Method | SFT | RL | Reward | Compiler-OPD | Error-Branch |
|--------|-----|-----|--------|-------------|-------------|
| B0 Zero-shot | | | | | |
| B1 SFT | ✅ | | | | |
| B2 GRPO-Outcome | ✅ | GRPO | compile 0/1 | | |
| B3 GRPO-Multi | ✅ | GRPO | 4× binary | | |
| B4 GSPO-Multi | ✅ | GSPO | 4× binary | | |
| M1 +OPD | ✅ | GSPO | 4× binary | ✅ | |
| M2 +Branch | ✅ | GSPO | 4× binary | | ✅ |
| **M3 Full** | ✅ | GSPO | 4× binary | ✅ | ✅ |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run baseline
python experiments/run.py --config configs/b2_grpo_outcome.yaml

# Run full method
python experiments/run.py --config configs/m3_full.yaml

# Colab
# See notebooks/colab_quickstart.ipynb
```

## Project Structure

```
├── configs/              # Experiment configurations
├── data/                 # Training data & prompts
│   ├── prepare.py        # Data preparation scripts
│   └── prompts/          # Generated prompts
├── environments/         # Coding environment
│   ├── horizon_env.py    # Horizon theme environment
│   └── validators/       # Theme check, schema, lint
├── rewards/              # Reward functions
│   ├── outcome.py        # Binary compile pass/fail
│   ├── multi_signal.py   # 4× binary rewards
│   ├── compiler_opd.py   # Compiler-Guided OPD
│   └── process.py        # Step-level process reward
├── exploration/          # Exploration strategies
│   ├── standard.py       # Temperature sampling
│   ├── dynamic.py        # DAPO dynamic sampling
│   └── error_branch.py   # Error-Triggered Branching
├── algorithms/           # RL algorithms
│   ├── grpo.py
│   └── gspo.py
├── experiments/          # Experiment runners
│   ├── run.py            # Main entry point
│   └── analysis.py       # Result analysis
├── evaluation/           # Metrics & analysis
│   ├── metrics.py
│   └── visualize.py      # Plots & figures
├── notebooks/            # Jupyter notebooks
│   └── colab_quickstart.ipynb
├── paper/                # LaTeX source
└── scripts/              # Utility scripts
```

## Model

- Primary: `Qwen/Qwen2.5-Coder-3B-Instruct` + QLoRA 4-bit
- Scaling: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Training: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl)

## Citation

```bibtex
@article{fu2026compiler,
  title={Compiler-as-Reward: Process Feedback for Coding Agent RL Training},
  author={Fu, Junhao},
  year={2026}
}
```

## License

MIT
