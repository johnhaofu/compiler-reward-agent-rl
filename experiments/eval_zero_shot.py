"""
B0: Zero-shot baseline evaluation.
No training — just load model and generate, then validate.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.sitemuse_validator import SitemuseValidator


def run_zero_shot(
    model_path: str,
    data_path: str = "data/prompts/val.jsonl",
    max_samples: int = 50,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    output_path: str = "experiments/results/b0_zero_shot.json",
):
    # 1. Load model
    print("Loading model...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model.eval()

    import torch
    vram = round(torch.cuda.memory_allocated() / 1024**3, 1)
    print(f"Model loaded. VRAM: {vram} GB")

    # 2. Load prompts
    prompts = []
    with open(data_path) as f:
        for line in f:
            prompts.append(json.loads(line))
            if len(prompts) >= max_samples:
                break
    print(f"Loaded {len(prompts)} prompts from {data_path}")

    # 3. Init validator
    validator = SitemuseValidator()

    # 4. Generate and validate
    results = []
    pass_count = 0
    json_pass = 0
    api_pass = 0

    for i, item in enumerate(prompts):
        messages = item["prompt"]
        tpl_type = item["template_type"]

        # Generate
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
            )
        gen_time = time.time() - t0

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Validate
        val_result = validator.validate(tpl_type, response)

        if val_result.json_valid:
            json_pass += 1
        if val_result.api_valid:
            api_pass += 1
        if val_result.all_passed:
            pass_count += 1

        result = {
            "index": i,
            "template_type": tpl_type,
            "industry": item.get("industry", ""),
            "complexity": item.get("complexity", ""),
            "json_valid": val_result.json_valid,
            "api_valid": val_result.api_valid,
            "all_passed": val_result.all_passed,
            "error": val_result.get_error_message(),
            "response_len": len(response),
            "gen_time": round(gen_time, 2),
        }
        results.append(result)

        status = "✅" if val_result.all_passed else "❌"
        print(f"  [{i+1}/{len(prompts)}] {status} {tpl_type:20s} | "
              f"json={val_result.json_valid} api={val_result.api_valid} | "
              f"{len(response)} chars {gen_time:.1f}s"
              + (f" | {val_result.get_error_message()[:80]}" if not val_result.all_passed else ""))

    # 5. Summary
    total = len(results)
    print(f"\n{'='*60}")
    print(f"B0 Zero-Shot Results ({model_path})")
    print(f"{'='*60}")
    print(f"Total samples:    {total}")
    print(f"JSON valid:       {json_pass}/{total} ({json_pass*100//total}%)")
    print(f"API valid:        {api_pass}/{total} ({api_pass*100//total}%)")
    print(f"All passed:       {pass_count}/{total} ({pass_count*100//total}%)")
    print(f"Avg gen time:     {sum(r['gen_time'] for r in results)/total:.1f}s")
    print(f"Avg response len: {sum(r['response_len'] for r in results)//total} chars")

    # By complexity
    for comp in ["low", "medium", "high"]:
        comp_results = [r for r in results if r["complexity"] == comp]
        if comp_results:
            cp = sum(1 for r in comp_results if r["all_passed"])
            print(f"  {comp}: {cp}/{len(comp_results)} ({cp*100//len(comp_results)}%)")

    # Error analysis
    errors = [r["error"] for r in results if r["error"]]
    if errors:
        print(f"\nTop errors:")
        from collections import Counter
        for err, cnt in Counter(errors).most_common(5):
            print(f"  {cnt}x: {err[:100]}")

    # 6. Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model_path,
        "experiment": "B0_zero_shot",
        "total": total,
        "json_pass_rate": json_pass / total,
        "api_pass_rate": api_pass / total,
        "all_pass_rate": pass_count / total,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/root/autodl-tmp/models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--data-path", type=str, default="data/prompts/val.jsonl")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="experiments/results/b0_zero_shot.json")
    args = parser.parse_args()

    run_zero_shot(
        model_path=args.model_path,
        data_path=args.data_path,
        max_samples=args.max_samples,
        output_path=args.output,
    )
