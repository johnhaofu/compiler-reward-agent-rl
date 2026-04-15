"""
Qwen2.5-Coder Agent Loop Baseline.

Same tools, same prompts, same validation as Claude baseline.
Uses Qwen's native function calling via vLLM.
"""

import json
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.tools import TOOLS, SYSTEM_PROMPT, AgentWorkspace
from evaluation.verify_engine import verify_episode
from experiments.notify import experiment_start, experiment_done, experiment_error


def build_qwen_tools(tools: list) -> list:
    """Convert our tool definitions to Qwen/OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
        }
        for t in tools
    ]


QWEN_TOOLS = build_qwen_tools(TOOLS)


def run_episode(
    model,
    tokenizer,
    workspace: AgentWorkspace,
    task_prompt: str,
    max_turns: int = 50,
    temperature: float = 0.7,
    max_new_tokens: int = 4096,
) -> dict:
    """Run one agent episode with Qwen model."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    turn = 0

    while turn < max_turns and not workspace.is_done:
        # Build prompt with tool definitions
        text = tokenizer.apply_chat_template(
            messages,
            tools=QWEN_TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        total_input_tokens += input_len

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
            )

        output_ids = outputs[0][input_len:]
        total_output_tokens += len(output_ids)
        response_text = tokenizer.decode(output_ids, skip_special_tokens=False)

        # Parse tool calls from response
        tool_calls = _parse_tool_calls(response_text)

        if not tool_calls:
            # No tool calls — model gave a text response, treat as done
            # Add the response and break
            clean_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": clean_text})
            break

        # Execute tool calls
        # Add assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": response_text}
        messages.append(assistant_msg)

        for tc in tool_calls:
            tool_name = tc["name"]
            arguments = tc["arguments"]

            result = workspace.execute_tool(
                turn=turn,
                tool_name=tool_name,
                arguments=arguments,
            )

            # Add tool result message
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": result,
            })

            # Progress
            args_str = json.dumps(arguments, ensure_ascii=False)[:60]
            result_preview = result[:80] + "..." if len(result) > 80 else result
            print(f"    [{turn}] {tool_name}({args_str}) → {result_preview}")

        turn += 1

    # Metrics
    metrics = workspace.get_metrics()
    metrics["input_tokens"] = total_input_tokens
    metrics["output_tokens"] = total_output_tokens
    metrics["total_tokens"] = total_input_tokens + total_output_tokens
    metrics["cost_usd"] = 0.0  # Local model, no API cost
    return metrics


def _parse_tool_calls(response_text: str) -> list[dict]:
    """
    Parse Qwen's tool call format from response text.

    Qwen uses:
      <tool_call>
      {"name": "tool_name", "arguments": {...}}
      </tool_call>

    Or the function_call format:
      ✿FUNCTION✿: tool_name
      ✿ARGS✿: {"key": "value"}

    Also handles standard JSON function call blocks.
    """
    import re

    tool_calls = []

    # Pattern 1: <tool_call> tags
    pattern1 = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response_text, re.DOTALL)
    for match in pattern1:
        try:
            tc = json.loads(match)
            if "name" in tc:
                args = tc.get("arguments", tc.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append({"name": tc["name"], "arguments": args})
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Pattern 2: Qwen function call format
    pattern2 = re.findall(r'✿FUNCTION✿:\s*(\w+)\s*\n✿ARGS✿:\s*(\{.*?\})', response_text, re.DOTALL)
    for name, args_str in pattern2:
        try:
            args = json.loads(args_str)
            tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Pattern 3: {"name": "...", "arguments": {...}} anywhere in text
    pattern3 = re.findall(r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{.*?\})\}', response_text, re.DOTALL)
    for name, args_str in pattern3:
        try:
            args = json.loads(args_str)
            tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            continue

    return tool_calls


def run_evaluation(
    model_path: str = "/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct",
    data_path: str = "data/prompts/eval_fixed.jsonl",
    horizon_path: str = "/root/autodl-tmp/horizon",
    max_samples: int = 999,
    max_turns: int = 50,
    temperature: float = 0.7,
    output_path: str = "experiments/results/qwen_baseline.json",
    run_id: int = 1,
):
    try:
        # 1. Load model
        print("Loading model...")
        os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=8192,
            load_in_4bit=True,
        )
        model.eval()

        import torch
        vram = round(torch.cuda.memory_allocated() / 1024**3, 1)
        print(f"Model loaded. VRAM: {vram} GB")

        # 2. Load eval set
        items = []
        with open(data_path) as f:
            for line in f:
                items.append(json.loads(line))
                if len(items) >= max_samples:
                    break
        print(f"Loaded {len(items)} tasks")

        model_name = model_path.split("/")[-1]
        experiment_start(f"Qwen B0 run{run_id}", model_name, len(items))

        # 3. Run episodes
        episodes = []

        for i, item in enumerate(items):
            task_id = item.get("id", f"task-{i}")
            level = item.get("level", 0)
            level_name = item.get("level_name", "")
            task_prompt = item["prompt"][-1]["content"]

            workspace = AgentWorkspace(
                horizon_path=horizon_path,
                components_path="data/horizon_components.json",
                schemas_dir="data/schemas",
            )

            print(f"\n[{i+1}/{len(items)}] {task_id} | L{level} {level_name}")

            t0 = time.time()
            metrics = run_episode(
                model=model,
                tokenizer=tokenizer,
                workspace=workspace,
                task_prompt=task_prompt,
                max_turns=max_turns,
                temperature=temperature,
            )
            metrics["wall_time"] = round(time.time() - t0, 2)
            metrics["task_id"] = task_id
            metrics["level"] = level
            metrics["level_name"] = level_name

            # Verify
            verify_result = verify_episode(item, metrics)
            metrics["verify_passed"] = verify_result["passed"]
            metrics["verify_total"] = verify_result["total_checks"]
            metrics["verify_passed_count"] = verify_result["passed_checks"]
            metrics["verify_failed"] = verify_result["failed_checks"]
            metrics["fully_resolved"] = metrics["resolved"] and verify_result["passed"]

            episodes.append(metrics)

            status = "✅" if metrics["fully_resolved"] else "❌"
            ftv = "1st✓" if metrics["first_try_valid"] else "fix" if metrics["fix_rate"] else "fail"
            verify_status = "✓" if verify_result["passed"] else f"✗{verify_result['passed_checks']}/{verify_result['total_checks']}"
            print(f"  {status} API [{ftv}] | verify:{verify_status} | "
                  f"turns={metrics['total_turns']} {metrics['wall_time']:.1f}s")
            if verify_result["failed_checks"]:
                for fc in verify_result["failed_checks"][:2]:
                    print(f"    verify fail: {fc['reason'][:80]}")

            workspace.cleanup()

        # 4. Summary
        total = len(episodes)
        resolved = sum(1 for e in episodes if e["resolved"])
        fully_resolved = sum(1 for e in episodes if e.get("fully_resolved", False))
        first_try = sum(1 for e in episodes if e["first_try_valid"])
        fixed = sum(1 for e in episodes if e["fix_rate"])
        verify_passed = sum(1 for e in episodes if e.get("verify_passed", False))
        resolved_eps = [e for e in episodes if e["resolved"]]

        print(f"\n{'='*70}")
        print(f"  Qwen Baseline Results — {model_name} (run {run_id})")
        print(f"{'='*70}")
        print(f"\n┌{'─'*50}┐")
        print(f"│ {'Metric':<30} {'Value':>18} │")
        print(f"├{'─'*50}┤")
        print(f"│ {'Total tasks':<30} {total:>18} │")
        print(f"│ {'API Resolved':<30} {f'{resolved}/{total} ({resolved*100//total}%)':>18} │")
        print(f"│ {'  ├ First-try Valid':<30} {f'{first_try}/{total} ({first_try*100//total}%)':>18} │")
        print(f"│ {'  └ Fixed after Error':<30} {f'{fixed}/{total} ({fixed*100//total}%)':>18} │")
        print(f"│ {'Verify Passed':<30} {f'{verify_passed}/{total} ({verify_passed*100//total}%)':>18} │")
        print(f"│ {'★ Fully Resolved':<30} {f'{fully_resolved}/{total} ({fully_resolved*100//total}%)':>18} │")
        avg_turns = sum(e["total_turns"] for e in episodes) / total
        avg_time = sum(e["wall_time"] for e in episodes) / total
        print(f"├{'─'*50}┤")
        print(f"│ {'Avg turns / task':<30} {avg_turns:>18.1f} │")
        print(f"│ {'Avg time / task':<30} {f'{avg_time:.1f}s':>18} │")
        print(f"└{'─'*50}┘")

        # By level
        print(f"\n┌{'─'*82}┐")
        print(f"│ {'Level':<24} {'API OK':>8} {'Verify':>8} {'★Full':>8} {'1st✓':>8} {'Fix':>8} {'Turns':>8} │")
        print(f"├{'─'*82}┤")
        for lv in sorted(set(e["level"] for e in episodes)):
            lv_eps = [e for e in episodes if e["level"] == lv]
            n = len(lv_eps)
            lv_name = lv_eps[0].get("level_name", "")
            print(f"│ L{lv} {lv_name:<21} "
                  f"{f'{sum(1 for e in lv_eps if e[\"resolved\"])}/{n}':>8} "
                  f"{f'{sum(1 for e in lv_eps if e.get(\"verify_passed\"))}/{n}':>8} "
                  f"{f'{sum(1 for e in lv_eps if e.get(\"fully_resolved\"))}/{n}':>8} "
                  f"{f'{sum(1 for e in lv_eps if e[\"first_try_valid\"])}/{n}':>8} "
                  f"{f'{sum(1 for e in lv_eps if e[\"fix_rate\"])}/{n}':>8} "
                  f"{sum(e['total_turns'] for e in lv_eps)/n:>8.1f} │")
        print(f"└{'─'*82}┘")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model_path,
            "experiment": f"qwen_baseline_run{run_id}",
            "total": total,
            "resolved_rate": resolved / total,
            "first_try_valid_rate": first_try / total,
            "fix_rate": fixed / total,
            "verify_pass_rate": verify_passed / total,
            "fully_resolved_rate": fully_resolved / total,
            "avg_turns": avg_turns,
            "avg_time": avg_time,
            "episodes": episodes,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

        experiment_done(f"Qwen B0 run{run_id}", summary)

    except Exception as e:
        experiment_error(f"Qwen B0 run{run_id}", str(e))
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data-path", default="data/prompts/eval_fixed.jsonl")
    parser.add_argument("--horizon-path", default="/root/autodl-tmp/horizon")
    parser.add_argument("--max-samples", type=int, default=999)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-path", default="experiments/results/qwen_baseline.json")
    parser.add_argument("--run-id", type=int, default=1)
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        data_path=args.data_path,
        horizon_path=args.horizon_path,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        temperature=args.temperature,
        output_path=args.output_path,
        run_id=args.run_id,
    )
