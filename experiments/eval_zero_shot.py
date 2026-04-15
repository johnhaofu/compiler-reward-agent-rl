"""
B0: Zero-shot baseline evaluation.
No training — just load model and generate via vLLM, then validate.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.sitemuse_validator import SitemuseValidator
from experiments.notify import experiment_start, experiment_done, experiment_error


def run_zero_shot(
    model_path: str,
    data_path: str = "data/prompts/val.jsonl",
    max_samples: int = 50,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    output_path: str = "experiments/results/b0_zero_shot.json",
):
    try:
        # 1. Load model with vLLM inference
        print("Loading model with vLLM...")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=4096,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.8,
        )

        import torch
        vram = round(torch.cuda.memory_allocated() / 1024**3, 1)
        print(f"Model loaded. VRAM: {vram} GB")

        # 2. Load prompts
        prompts_data = []
        with open(data_path) as f:
            for line in f:
                prompts_data.append(json.loads(line))
                if len(prompts_data) >= max_samples:
                    break
        print(f"Loaded {len(prompts_data)} prompts from {data_path}")

        # 3. Init validator
        validator = SitemuseValidator()

        # 4. Prepare all prompts for batch generation
        experiment_start("B0 Zero-Shot", model_path.split("/")[-1], len(prompts_data))

        chat_inputs = []
        for item in prompts_data:
            messages = item["prompt"]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            chat_inputs.append(text)

        # 5. Batch generate with vLLM
        print(f"Generating {len(chat_inputs)} responses via vLLM...")
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
        )

        t0 = time.time()
        outputs = model.fast_generate(
            chat_inputs,
            sampling_params=sampling_params,
            lora_request=None,
        )
        total_gen_time = time.time() - t0
        print(f"Generation done in {total_gen_time:.1f}s ({total_gen_time/len(chat_inputs):.1f}s/sample)")

        # 6. Validate all responses
        print("Validating responses...")
        results = []
        pass_count = 0
        json_pass = 0
        api_pass = 0

        for i, (item, output) in enumerate(zip(prompts_data, outputs)):
            tpl_type = item["template_type"]
            response = output.outputs[0].text

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
                "response_preview": response[:200],
            }
            results.append(result)

            status = "✅" if val_result.all_passed else "❌"
            print(f"  [{i+1}/{len(prompts_data)}] {status} {tpl_type:20s} | "
                  f"json={val_result.json_valid} api={val_result.api_valid}"
                  + (f" | {val_result.get_error_message()[:80]}" if not val_result.all_passed else ""))

        # 7. Summary
        total = len(results)
        print(f"\n{'='*60}")
        print(f"B0 Zero-Shot Results ({model_path.split('/')[-1]})")
        print(f"{'='*60}")
        print(f"Total samples:    {total}")
        print(f"JSON valid:       {json_pass}/{total} ({json_pass*100//total}%)")
        print(f"API valid:        {api_pass}/{total} ({api_pass*100//total}%)")
        print(f"All passed:       {pass_count}/{total} ({pass_count*100//total}%)")
        print(f"Total gen time:   {total_gen_time:.1f}s ({total_gen_time/total:.1f}s/sample)")

        # By complexity
        for comp in ["low", "medium", "high"]:
            comp_results = [r for r in results if r["complexity"] == comp]
            if comp_results:
                cp = sum(1 for r in comp_results if r["all_passed"])
                print(f"  {comp}: {cp}/{len(comp_results)} ({cp*100//max(len(comp_results),1)}%)")

        # Error analysis
        errors = [r["error"] for r in results if r["error"]]
        if errors:
            print(f"\nTop errors:")
            from collections import Counter
            for err, cnt in Counter(errors).most_common(5):
                print(f"  {cnt}x: {err[:100]}")

        # 8. Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model_path,
            "experiment": "B0_zero_shot",
            "total": total,
            "json_pass_rate": json_pass / total,
            "api_pass_rate": api_pass / total,
            "all_pass_rate": pass_count / total,
            "gen_time_total": total_gen_time,
            "gen_time_per_sample": total_gen_time / total,
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

        experiment_done("B0 Zero-Shot", summary)

    except Exception as e:
        experiment_error("B0 Zero-Shot", str(e))
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/root/autodl-tmp/models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--data-path", type=str, default="data/prompts/val.jsonl")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="experiments/results/b0_zero_shot.json")
    args = parser.parse_args()

    run_zero_shot(
        model_path=args.model_path,
        data_path=args.data_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        output_path=args.output,
    )
