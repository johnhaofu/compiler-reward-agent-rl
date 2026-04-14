"""
Main experiment runner.

Usage:
    python experiments/run.py --config configs/b2_grpo_outcome.yaml
    python experiments/run.py --quick-test  # Quick smoke test
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def quick_test(horizon_path: str, model_path: str):
    """Quick smoke test: load model, validate one template, run 10 steps."""
    from environments.horizon_env import HorizonEnvironment

    # 1. Test environment
    print("=" * 60)
    print("Step 1: Testing HorizonEnvironment")
    print("=" * 60)
    env = HorizonEnvironment(horizon_path)
    print(f"  Sections: {len(env.available_sections)}")
    print(f"  Blocks: {len(env.available_blocks)}")
    print(f"  Templates: {env.list_templates()}")

    # Validate existing templates
    for tpl in env.list_templates():
        gt = env.get_ground_truth(tpl)
        if gt:
            result = env.validate(tpl, gt)
            status = "✅" if result.all_passed else "❌"
            print(f"  {status} {tpl}")
            if not result.all_passed:
                print(f"     {result.get_error_message()}")

    # 2. Test model loading
    print("\n" + "=" * 60)
    print("Step 2: Loading model")
    print("=" * 60)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    import torch
    vram = round(torch.cuda.memory_allocated() / 1024**3, 1)
    print(f"  Model loaded! VRAM: {vram} GB")

    # 3. Test generation
    print("\n" + "=" * 60)
    print("Step 3: Test generation")
    print("=" * 60)
    prompt = """Generate a Shopify Horizon theme template JSON for a simple "About Us" page.
The page should have:
- A heading section with the page title
- A text section with page content

Output ONLY valid JSON, no comments, no explanation."""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Generated ({len(response)} chars):")
    print(f"  {response[:500]}...")

    # 4. Validate generation
    print("\n" + "=" * 60)
    print("Step 4: Validate generation")
    print("=" * 60)
    result = env.validate("page", response)
    print(f"  JSON valid:        {result.json_valid}")
    print(f"  Sections valid:    {result.sections_valid}")
    print(f"  Theme check valid: {result.theme_check_valid}")
    if not result.all_passed:
        print(f"  Error: {result.get_error_message()}")

    # 5. Test reward functions
    print("\n" + "=" * 60)
    print("Step 5: Test reward functions")
    print("=" * 60)
    from rewards.outcome import outcome_reward
    from rewards.multi_signal import multi_signal_reward

    completions = [[{"content": response}]]
    prompts_list = [messages]

    r_outcome = outcome_reward(
        prompts_list, completions,
        env=env, template_types=["page"]
    )
    print(f"  Outcome reward:      {r_outcome}")

    r_multi = multi_signal_reward(
        prompts_list, completions,
        env=env, template_types=["page"]
    )
    print(f"  Multi-signal reward: {r_multi}")

    # 6. Test OPD hint extraction
    print("\n" + "=" * 60)
    print("Step 6: Test Compiler-OPD hint extraction")
    print("=" * 60)
    from rewards.compiler_opd import extract_compiler_hints

    if not result.all_passed:
        hints = extract_compiler_hints(result, prompt)
        print(f"  Extracted {len(hints)} hints:")
        for h in hints:
            print(f"    [{h.source}] {h.message[:100]}")
    else:
        print("  Generation passed all checks — no hints needed")

    print("\n" + "=" * 60)
    print("✅ Quick test complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compiler-Reward Agent RL Experiments")
    parser.add_argument("--config", type=str, help="Path to experiment config YAML")
    parser.add_argument("--quick-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--horizon-path", type=str, default="/root/autodl-tmp/horizon",
                        help="Path to Horizon theme")
    parser.add_argument("--model-path", type=str,
                        default="/root/autodl-tmp/models/Qwen3-Coder-30B-A3B-Instruct",
                        help="Path to model")
    args = parser.parse_args()

    if args.quick_test:
        quick_test(args.horizon_path, args.model_path)
    elif args.config:
        print(f"Running experiment from config: {args.config}")
        # TODO: Implement config-based experiment runner
        raise NotImplementedError("Config-based runner coming next")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
