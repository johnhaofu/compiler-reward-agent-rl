"""
GRPO Training with multi-GPU support (no Unsloth dependency).

Uses HuggingFace Transformers + PEFT LoRA + TRL GRPOTrainer + accelerate.
Supports 3-GPU training via FSDP/DDP.

Usage:
  # Multi-GPU (3 GPUs)
  accelerate launch --num_processes 3 training/grpo_multigpu.py \
    --model-name /root/autodl-tmp/models/Qwen3.5-4B \
    --horizon-path /root/autodl-tmp/horizon \
    --reward-type multi_signal \
    --max-completion-len 4096

  # Single GPU
  python training/grpo_multigpu.py --max-completion-len 2048
"""

import json
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable TRL's vLLM backend
import importlib
_trl_utils = importlib.import_module("trl.import_utils")
_trl_utils.is_vllm_available = lambda: False


SYSTEM_PROMPT = """You are an expert Shopify theme developer working with the Horizon theme.
Generate a valid Shopify page template as JSON. The JSON must have "sections" and "order" keys.
Only use section types that exist in the Horizon theme. Output valid JSON only, no markdown or comments."""


def load_prompts(data_path: str) -> list[dict]:
    items = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            user_content = item["prompt"][-1]["content"]
            prompt_text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            items.append({
                "prompt": prompt_text,
                "template_type": item.get("template_type", "page"),
            })
    return items


def main():
    parser = argparse.ArgumentParser(description="GRPO Multi-GPU Training")
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/Qwen3.5-4B")
    parser.add_argument("--data-path", default="data/prompts/train.jsonl")
    parser.add_argument("--horizon-path", default="/root/autodl-tmp/horizon")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/models/qwen3.5-grpo-b3")
    parser.add_argument("--reward-type", choices=["outcome", "multi_signal"], default="multi_signal")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=999)
    args = parser.parse_args()

    print(f"Model: {args.model_name}")
    print(f"Reward: {args.reward_type}")

    # ── Load model with standard HF + PEFT ──
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load environment ──
    from environments.horizon_env import HorizonEnvironment
    env = HorizonEnvironment(args.horizon_path)
    print(f"Horizon: {len(env.available_sections)} sections, {len(env.available_blocks)} blocks")

    # ── Load prompts ──
    all_prompts = load_prompts(args.data_path)[:args.max_samples]
    template_types_map = {p["prompt"]: p["template_type"] for p in all_prompts}
    print(f"Training prompts: {len(all_prompts)}")

    from datasets import Dataset
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in all_prompts])

    # ── Reward function ──
    def grpo_reward_func(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            tpl_type = template_types_map.get(prompt, "page")
            try:
                result = env.validate(tpl_type, text)
                if args.reward_type == "outcome":
                    rewards.append(1.0 if result.all_passed else 0.0)
                else:
                    rd = result.to_reward_dict()
                    score = (rd["json_valid"] + rd["sections_valid"] + rd["theme_check_valid"]) / 3.0
                    if result.all_passed:
                        score = 1.0
                    rewards.append(score)
            except Exception:
                rewards.append(0.0)
        return rewards

    # ── GRPO Training ──
    from trl import GRPOTrainer, GRPOConfig

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        generation_kwargs={"temperature": args.temperature, "do_sample": True},
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=grpo_reward_func,
    )

    print(f"Starting GRPO training ({args.reward_type} reward, multi-GPU)...")
    trainer.train()

    # ── Save ──
    print(f"Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = vars(args)
    config["num_prompts"] = len(all_prompts)
    with open(Path(args.output_dir) / "grpo_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("GRPO training complete!")


if __name__ == "__main__":
    main()
