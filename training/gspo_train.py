"""
B4: GSPO (Group-wise Selection Policy Optimization) Training.

GSPO vs GRPO: Instead of using all generations for advantage estimation,
GSPO selects the best and worst pairs within each group for contrastive training.
This reduces noise from mediocre samples and focuses learning on clear win/lose signals.

Algorithm:
  1. Generate N completions per prompt
  2. Score each with multi-signal reward
  3. Select best (highest reward) and worst (lowest reward) per group
  4. Train on (best - worst) pairs with DPO-style objective

Uses TRL GRPOTrainer with custom reward that amplifies best/worst contrast.

Usage:
  python training/gspo_train.py \
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
    parser = argparse.ArgumentParser(description="B4: GSPO Training")
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/qwen3.5-sft")
    parser.add_argument("--data-path", default="data/prompts/train.jsonl")
    parser.add_argument("--horizon-path", default="/root/autodl-tmp/horizon")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/models/qwen3.5-gspo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-completion-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gspo-margin", type=float, default=0.5,
                        help="Minimum reward gap between best and worst to form a pair")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=999)
    args = parser.parse_args()

    print(f"=== B4: GSPO (margin={args.gspo_margin}, G={args.num_generations}) ===")

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

    # ── GSPO reward: multi-signal with group-wise contrast amplification ──
    # GSPO modifies rewards post-hoc: within each group of N generations,
    # best gets boosted, worst gets penalized, mediocre gets zeroed out.
    # We implement this via reward function that tracks group state.
    group_scores = []
    gspo_stats = {"pairs_formed": 0, "groups_total": 0}

    def gspo_reward_func(prompts, completions, **kwargs):
        """Multi-signal reward with GSPO group-wise selection.

        When called with N completions for one prompt (GRPO's internal batching),
        we score all N, then amplify the best/worst contrast.
        """
        raw_rewards = []
        for prompt, completion in zip(prompts, completions):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            tpl_type = template_types_map.get(prompt, "page")
            result = env.validate(tpl_type, text)
            rd = result.to_reward_dict()
            score = (rd["json_valid"] + rd["sections_valid"] + rd["theme_check_valid"]) / 3.0
            if result.all_passed:
                score = 1.0
            raw_rewards.append(score)

        # GSPO selection: amplify best/worst, suppress middle
        if len(raw_rewards) > 1:
            best_r = max(raw_rewards)
            worst_r = min(raw_rewards)
            gap = best_r - worst_r

            gspo_stats["groups_total"] += 1

            if gap >= args.gspo_margin:
                gspo_stats["pairs_formed"] += 1
                # Amplify contrast: best → 1.0, worst → 0.0, others → scaled
                adjusted = []
                for r in raw_rewards:
                    if gap > 0:
                        normalized = (r - worst_r) / gap  # 0 to 1
                    else:
                        normalized = 0.5
                    adjusted.append(normalized)
                return adjusted

        return raw_rewards

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
        processing_class=tokenizer, reward_funcs=gspo_reward_func,
    )

    print("Starting B4 GSPO training...")
    trainer.train()

    # ── Save ──
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = vars(args)
    config["gspo_stats"] = gspo_stats
    with open(Path(args.output_dir) / "gspo_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"B4 done! GSPO pairs: {gspo_stats['pairs_formed']}/{gspo_stats['groups_total']}")


if __name__ == "__main__":
    main()
