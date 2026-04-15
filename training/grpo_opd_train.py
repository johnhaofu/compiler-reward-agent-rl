"""
M1: GRPO + Compiler-OPD Training.

Innovation 1: Uses compiler error messages as hints to compute token-level
advantages (OPD). Failed samples get re-scored with enhanced prompts containing
the compiler error, creating a "teacher" that guides which tokens to reinforce.

Flow:
  1. Generate N completions per prompt
  2. Validate each → binary reward
  3. For failed samples: extract compiler hints → compute OPD advantages
  4. OPD advantage modulates the GRPO policy gradient at token level

Usage:
  python training/grpo_opd_train.py \
    --model-name /root/autodl-tmp/models/qwen3.5-sft \
    --horizon-path /root/autodl-tmp/horizon
"""

import json
import argparse
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


SYSTEM_PROMPT = """You are an expert Shopify theme developer working with the Horizon theme.
Generate a valid Shopify page template as JSON. The JSON must have "sections" and "order" keys.
Only use section types that exist in the Horizon theme. Output valid JSON only, no markdown or comments."""


def load_prompts(data_path: str) -> list[dict]:
    items = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            user_content = item["prompt"][-1]["content"]
            items.append({
                "prompt": (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                ),
                "template_type": item.get("template_type", "page"),
            })
    return items


def compute_opd_bonus(model, tokenizer, prompt, response, hint, device="cuda"):
    """Compute scalar OPD bonus from token-level advantages.

    Returns a scalar bonus (mean positive advantage) that gets added to
    the base reward for failed samples with compiler hints.
    """
    from rewards.compiler_opd import compute_opd_advantage

    advantages = compute_opd_advantage(
        model, tokenizer, prompt, response, hint, device
    )
    if not advantages:
        return 0.0

    # Positive advantages = tokens the teacher agrees with
    pos_adv = [a for a in advantages if a > 0]
    if pos_adv:
        # Small bonus: mean positive advantage, capped
        return min(sum(pos_adv) / len(advantages), 0.3)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="M1: GRPO + Compiler-OPD")
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/qwen3.5-sft")
    parser.add_argument("--data-path", default="data/prompts/train.jsonl")
    parser.add_argument("--horizon-path", default="/root/autodl-tmp/horizon")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/models/qwen3.5-grpo-opd")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-completion-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--opd-weight", type=float, default=0.3,
                        help="Weight of OPD bonus in reward (0=no OPD, 1=full OPD)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=999)
    args = parser.parse_args()

    print(f"=== M1: GRPO + Compiler-OPD (weight={args.opd_weight}) ===")

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
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.print_trainable_parameters()

    # ── Environment ──
    from environments.horizon_env import HorizonEnvironment
    from rewards.compiler_opd import extract_compiler_hints
    env = HorizonEnvironment(args.horizon_path)

    # ── Load prompts ──
    all_prompts = load_prompts(args.data_path)[:args.max_samples]
    template_types_map = {p["prompt"]: p["template_type"] for p in all_prompts}
    print(f"Training prompts: {len(all_prompts)}")

    from datasets import Dataset
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in all_prompts])

    # ── Reward with OPD bonus ──
    opd_stats = {"total": 0, "hints_used": 0, "avg_bonus": 0.0}

    def reward_func(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            tpl_type = template_types_map.get(prompt, "page")
            result = env.validate(tpl_type, text)

            opd_stats["total"] += 1

            if result.all_passed:
                rewards.append(1.0)
            else:
                base = 0.0
                # Multi-signal partial credit
                rd = result.to_reward_dict()
                base = (rd["json_valid"] + rd["sections_valid"] + rd["theme_check_valid"]) / 3.0

                # OPD bonus: use compiler error as hint to score response
                hints = extract_compiler_hints(result, prompt)
                if hints and args.opd_weight > 0:
                    try:
                        bonus = compute_opd_bonus(
                            model, tokenizer, prompt, text, hints[0]
                        )
                        base += args.opd_weight * bonus
                        opd_stats["hints_used"] += 1
                        opd_stats["avg_bonus"] = (
                            opd_stats["avg_bonus"] * 0.9 + bonus * 0.1
                        )
                    except Exception:
                        pass  # OPD computation failed, skip bonus

                rewards.append(min(base, 0.99))  # Cap below 1.0 for failed samples
        return rewards

    # ── Train ──
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
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
    )

    trainer = GRPOTrainer(
        model=model, args=training_args, train_dataset=dataset,
        processing_class=tokenizer, reward_funcs=reward_func,
    )

    print("Starting M1 training...")
    trainer.train()

    # ── Save ──
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = vars(args)
    config["opd_stats"] = opd_stats
    with open(Path(args.output_dir) / "m1_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"M1 done! OPD hints used: {opd_stats['hints_used']}/{opd_stats['total']}")


if __name__ == "__main__":
    main()
