# CLAUDE.md

## Project Overview

Research project: "Compiler-as-Reward: Process Feedback for Coding Agent RL Training"

Using compiler/linter feedback as reward signals for RL training of coding agents. Based on Shopify Horizon theme's Liquid template generation.

## Key Decisions

- Base model: Qwen2.5-Coder-3B-Instruct + QLoRA 4-bit
- Training: Unsloth + TRL GRPOTrainer
- Environment: Shopify Horizon theme, validated via `shopify theme check`
- Reward: Binary signals (pass/fail), not continuous scores

## Code Conventions

- Python 3.11+
- Type hints required
- All experiments reproducible via config YAML files
- Results saved as JSON for analysis
