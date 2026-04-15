"""
SFT Training for Qwen3.5-4B on Shopify Horizon agent trajectories.

Uses Unsloth + TRL SFTTrainer with QLoRA 4-bit.

Usage (on GPU server):
  python training/sft_train.py
  python training/sft_train.py --epochs 3 --lr 2e-4 --output-dir /root/autodl-tmp/models/qwen3.5-sft

Prerequisites:
  pip install unsloth trl datasets
"""

import json
import argparse
from pathlib import Path


def load_sft_dataset(data_path: str) -> list[dict]:
    """Load SFT trajectories from JSONL."""
    items = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            items.append(item)
    return items


def format_for_trl(items: list[dict]) -> list[dict]:
    """Convert to TRL SFTTrainer format.

    TRL expects {"messages": [...]} where messages follow OpenAI chat format.
    The model's chat template handles tool_calls formatting.
    """
    formatted = []
    for item in items:
        messages = item["messages"]
        # Filter: only keep well-formed conversations
        if len(messages) < 4:  # At minimum: system + user + assistant + tool
            continue
        # Ensure messages are clean
        clean_msgs = []
        for msg in messages:
            m = {"role": msg["role"]}
            if msg.get("content") is not None:
                m["content"] = msg["content"]
            else:
                m["content"] = ""
            if msg.get("tool_calls"):
                m["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                m["tool_call_id"] = msg["tool_call_id"]
            clean_msgs.append(m)
        formatted.append({"messages": clean_msgs})
    return formatted


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen3.5-4B")
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/Qwen3.5-4B")
    parser.add_argument("--data-path", default="data/sft/train.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/models/qwen3.5-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")

    # ── Load model with Unsloth ──
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # ── Apply LoRA ──
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print(f"LoRA params: r={args.lora_r}, alpha={args.lora_alpha}")
    model.print_trainable_parameters()

    # ── Load dataset ──
    raw_items = load_sft_dataset(args.data_path)
    formatted = format_for_trl(raw_items)
    print(f"Training examples: {len(formatted)}")

    from datasets import Dataset
    dataset = Dataset.from_list(formatted)

    # ── Train with TRL SFTTrainer ──
    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
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
        save_total_limit=3,
        bf16=True,
        max_seq_length=args.max_seq_len,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
    )

    def formatting_func(example):
        """Format a single example's messages using tokenizer's chat template."""
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    # ── Save ──
    print(f"Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config
    config = vars(args)
    config["num_examples"] = len(formatted)
    with open(Path(args.output_dir) / "sft_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("SFT training complete!")


if __name__ == "__main__":
    main()
