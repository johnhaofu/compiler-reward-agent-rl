"""
GRPO Training for Qwen3.5-4B on Shopify Horizon template generation.

Single-turn GRPO: prompt → generate template JSON → validate → reward.
Uses compiler feedback (JSON syntax + section/block reference check) as reward.

Experiments:
  B2: outcome_reward (binary pass/fail)
  B3: multi_signal_reward (json_valid + sections_valid + theme_check)

Usage (on GPU server):
  # B2: outcome reward
  python training/grpo_train.py \
    --model-name /root/autodl-tmp/models/qwen3.5-sft \
    --horizon-path /root/autodl-tmp/horizon \
    --reward-type outcome

  # B3: multi-signal reward
  python training/grpo_train.py \
    --model-name /root/autodl-tmp/models/qwen3.5-sft \
    --reward-type multi_signal
"""

import json
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_prompts(data_path: str, system_prompt: str) -> list[dict]:
    """Load training prompts formatted for GRPO.

    Returns list of {"prompt": str, "template_type": str}.
    The prompt is the full chat-formatted string.
    """
    items = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            user_content = item["prompt"][-1]["content"]
            template_type = item.get("template_type", "page")

            # Format as single-turn: system + user → model generates JSON
            prompt_text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            items.append({
                "prompt": prompt_text,
                "template_type": template_type,
            })
    return items


# ── Reward Functions ──

def make_outcome_reward(env):
    """Binary: 1.0 if validation passes, 0.0 otherwise."""
    def reward_func(completions, prompts, template_types, **kwargs):
        rewards = []
        for completion, tpl_type in zip(completions, template_types):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            result = env.validate(tpl_type, text)
            rewards.append(1.0 if result.all_passed else 0.0)
        return rewards
    return reward_func


def make_multi_signal_reward(env):
    """Sum of binary signals: json_valid + sections_valid + theme_check."""
    def reward_func(completions, prompts, template_types, **kwargs):
        rewards = []
        for completion, tpl_type in zip(completions, template_types):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            result = env.validate(tpl_type, text)
            rd = result.to_reward_dict()
            score = (rd["json_valid"] + rd["sections_valid"] + rd["theme_check_valid"]) / 3.0
            if result.all_passed:
                score = 1.0
            rewards.append(score)
        return rewards
    return reward_func


SYSTEM_PROMPT = """You are an expert Shopify theme developer working with the Horizon theme.
Generate a valid Shopify page template as JSON. The JSON must have "sections" and "order" keys.
Only use section types that exist in the Horizon theme. Output valid JSON only, no markdown or comments."""


def main():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/qwen3.5-sft")
    parser.add_argument("--data-path", default="data/prompts/train.jsonl")
    parser.add_argument("--horizon-path", default="/root/autodl-tmp/horizon")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/models/qwen3.5-grpo")
    parser.add_argument("--reward-type", choices=["outcome", "multi_signal"], default="outcome")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-completion-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=999)
    parser.add_argument("--use-vllm", action="store_true", default=False,
                        help="Use vLLM for fast generation")
    parser.add_argument("--vllm-server-url", type=str, default=None,
                        help="External vLLM server URL (e.g. http://localhost:8000/v1). If set, uses server mode.")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.3,
                        help="GPU memory fraction for vLLM colocate mode")
    args = parser.parse_args()

    print(f"Model: {args.model_name}")
    print(f"Reward: {args.reward_type}")
    print(f"Horizon: {args.horizon_path}")

    # ── Load model ──
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_prompt_len + args.max_completion_len,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    model.print_trainable_parameters()

    # ── Load environment ──
    from environments.horizon_env import HorizonEnvironment
    env = HorizonEnvironment(args.horizon_path)
    print(f"Horizon: {len(env.available_sections)} sections, {len(env.available_blocks)} blocks")

    # ── Select reward function ──
    if args.reward_type == "outcome":
        reward_fn = make_outcome_reward(env)
    else:
        reward_fn = make_multi_signal_reward(env)

    # ── Load prompts ──
    all_prompts = load_prompts(args.data_path, SYSTEM_PROMPT)
    if args.max_samples < len(all_prompts):
        all_prompts = all_prompts[:args.max_samples]
    print(f"Training prompts: {len(all_prompts)}")

    # Extract template_types for reward function
    template_types_map = {p["prompt"]: p["template_type"] for p in all_prompts}

    from datasets import Dataset
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in all_prompts])

    # ── Wrap reward for GRPOTrainer ──
    # GRPOTrainer calls reward_func(prompts=..., completions=...)
    def grpo_reward_func(prompts, completions, **kwargs):
        tpl_types = [template_types_map.get(p, "page") for p in prompts]
        # Extract text from completions
        texts = []
        for c in completions:
            if isinstance(c, list) and len(c) > 0:
                texts.append(c[0].get("content", "") if isinstance(c[0], dict) else str(c[0]))
            elif isinstance(c, str):
                texts.append(c)
            else:
                texts.append(str(c))

        rewards = []
        for text, tpl_type in zip(texts, tpl_types):
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
        # GRPO specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        generation_kwargs={"temperature": args.temperature, "do_sample": True},
        # vLLM acceleration
        use_vllm=args.use_vllm or bool(args.vllm_server_url),
        vllm_mode="server" if args.vllm_server_url else "colocate",
        vllm_server_base_url=args.vllm_server_url,
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=grpo_reward_func,
    )

    print(f"Starting GRPO training ({args.reward_type} reward)...")
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
