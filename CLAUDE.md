# CLAUDE.md

## Project Overview

Research project: "Compiler-as-Reward: Process Feedback for Coding Agent RL Training"

Using compiler/linter feedback as reward signals for RL training of coding agents. Based on Shopify Horizon theme's Liquid template generation.

## Key Decisions

- Base model: Qwen3.5-4B (hybrid GatedDeltaNet+MoE, text-only mode)
- Baselines: Claude Sonnet 4 (upper bound), Qwen2.5-Coder-7B (weak baseline)
- Training: Unsloth + TRL GRPOTrainer
- Inference: vLLM nightly with --tool-call-parser qwen3_coder --reasoning-parser qwen3
- Environment: Shopify Horizon theme, validated via Sitemuse API
- Reward: Binary signals (pass/fail), not continuous scores

## Code Conventions

- Python 3.11+
- Type hints required
- All experiments reproducible via config YAML files
- Results saved as JSON for analysis
