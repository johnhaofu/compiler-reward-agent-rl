"""
M2: GSPO + Error-Triggered Branching.

Innovation 2: When compilation fails, use the error message to create a
branching point — generate alternative completions conditioned on the error.
This adaptively explores the solution space around failure points.

Algorithm:
  1. Generate N completions per prompt
  2. Validate each
  3. For failed completions: create branch prompt = original + error message
  4. Generate M more completions from branch prompt
  5. Validate branch completions
  6. Include successful branches in the reward pool (bonus for recovery)

Usage:
  python training/error_branch_train.py \
    --model-name /root/autodl-tmp/models/qwen3.5-sft \
    --horizon-path /root/autodl-tmp/horizon
"""

import json
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="M2: GSPO + Error-Branch")
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/qwen3.5-sft")
    parser.add_argument("--data-path", default="data/prompts/train.jsonl")
    parser.add_argument("--horizon-path", default="/root/autodl-tmp/horizon")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/models/qwen3.5-error-branch")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--branch-generations", type=int, default=2,
                        help="Extra generations per failed sample using error hint")
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-completion-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--branch-temperature", type=float, default=0.5,
                        help="Lower temp for branch attempts (more focused)")
    parser.add_argument("--recovery-bonus", type=float, default=0.2,
                        help="Bonus reward for successful branch recovery")
    parser.add_argument("--gspo-margin", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=999)
    args = parser.parse_args()

    print(f"=== M2: GSPO + Error-Branch (branch_gen={args.branch_generations}) ===")

    # ── Load model ──
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_prompt_len + args.max_completion_len,
        load_in_4bit=True, dtype=None,
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
    env = HorizonEnvironment(args.horizon_path)

    # ── Load prompts ──
    all_prompts = load_prompts(args.data_path)[:args.max_samples]
    template_types_map = {p["prompt"]: p["template_type"] for p in all_prompts}
    print(f"Training prompts: {len(all_prompts)}")

    from datasets import Dataset
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in all_prompts])

    # ── Error-Branch reward ──
    branch_stats = {"total": 0, "branches_triggered": 0, "branches_recovered": 0}

    def error_branch_reward(prompts, completions, **kwargs):
        """GSPO reward with error-triggered branching.

        For each failed completion:
        1. Extract compiler error
        2. Create branch prompt: original + "[Previous attempt failed: {error}]\nPlease fix:"
        3. Generate branch completions (via model.generate)
        4. If branch succeeds: original gets partial credit, signals recovery path

        Since TRL's GRPOTrainer controls generation, we implement branching
        in the reward function by re-generating for failed samples.
        """
        rewards = []

        for prompt, completion in zip(prompts, completions):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            tpl_type = template_types_map.get(prompt, "page")
            result = env.validate(tpl_type, text)

            branch_stats["total"] += 1

            if result.all_passed:
                rewards.append(1.0)
                continue

            # Base multi-signal score
            rd = result.to_reward_dict()
            base = (rd["json_valid"] + rd["sections_valid"] + rd["theme_check_valid"]) / 3.0

            # Error-Branch: try to recover using error message
            error_msg = result.get_error_message()
            if error_msg and len(error_msg) > 10:
                branch_stats["branches_triggered"] += 1

                # Create branch prompt with error hint
                branch_prompt = (
                    f"{prompt}"
                    f"[Previous attempt failed with error: {error_msg}]\n"
                    f"Please generate a corrected version:\n"
                )

                try:
                    # Generate branch completions
                    branch_inputs = tokenizer(
                        branch_prompt, return_tensors="pt", truncation=True,
                        max_length=args.max_prompt_len
                    ).to(model.device)

                    branch_outputs = model.generate(
                        **branch_inputs,
                        max_new_tokens=args.max_completion_len,
                        temperature=args.branch_temperature,
                        do_sample=True,
                        num_return_sequences=args.branch_generations,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                    # Check if any branch succeeds
                    for output in branch_outputs:
                        branch_text = tokenizer.decode(
                            output[branch_inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True
                        )
                        branch_result = env.validate(tpl_type, branch_text)
                        if branch_result.all_passed:
                            # Recovery success! Give bonus
                            base += args.recovery_bonus
                            branch_stats["branches_recovered"] += 1
                            break
                        elif branch_result.json_valid and not result.json_valid:
                            # Partial improvement
                            base += args.recovery_bonus * 0.5
                            break
                except Exception:
                    pass  # Branch generation failed, keep base reward

            # GSPO-style: cap failed samples below 1.0
            rewards.append(min(base, 0.99))

        # GSPO group selection
        if len(rewards) > 1:
            best_r = max(rewards)
            worst_r = min(rewards)
            gap = best_r - worst_r
            if gap >= args.gspo_margin:
                return [(r - worst_r) / gap if gap > 0 else 0.5 for r in rewards]

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
        warmup_steps=5, weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=2,
        bf16=True, seed=args.seed,
        report_to="none", gradient_checkpointing=True,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
    )

    trainer = GRPOTrainer(
        model=model, args=training_args, train_dataset=dataset,
        processing_class=tokenizer, reward_funcs=error_branch_reward,
    )

    print("Starting M2 training...")
    trainer.train()

    # ── Save ──
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = vars(args)
    config["branch_stats"] = branch_stats
    with open(Path(args.output_dir) / "m2_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"M2 done! Branches: {branch_stats['branches_triggered']}, "
          f"Recovered: {branch_stats['branches_recovered']}/{branch_stats['total']}")


if __name__ == "__main__":
    main()
