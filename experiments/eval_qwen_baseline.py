"""
Qwen2.5-Coder Agent Loop Baseline via vLLM OpenAI-compatible API.

Start vLLM server first:
  python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
    --dtype half --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes

Then run:
  python experiments/eval_qwen_baseline.py
"""

import json
import sys
import time
import os
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.tools import TOOLS, SYSTEM_PROMPT, AgentWorkspace
from evaluation.verify_engine import verify_episode
from experiments.notify import experiment_start, experiment_done, experiment_error


def build_openai_tools(tools: list) -> list:
    """Our tool defs are already OpenAI format."""
    return tools


def run_episode(
    client: OpenAI,
    model_name: str,
    workspace: AgentWorkspace,
    task_prompt: str,
    max_turns: int = 50,
    temperature: float = 0.7,
) -> dict:
    """Run one agent episode via OpenAI-compatible API (vLLM)."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    turn = 0

    while turn < max_turns and not workspace.is_done:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=4096,
            )
        except Exception as e:
            print(f"    API error: {e}")
            break

        choice = response.choices[0]
        message = choice.message
        total_input_tokens += response.usage.prompt_tokens if response.usage else 0
        total_output_tokens += response.usage.completion_tokens if response.usage else 0

        # Append assistant message
        messages.append(message.model_dump())

        # Check for tool calls
        if not message.tool_calls:
            # Text response, no tool calls — done
            break

        # Execute tool calls
        for tc in message.tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            result = workspace.execute_tool(
                turn=turn,
                tool_name=tool_name,
                arguments=arguments,
            )

            # Add tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

            args_str = json.dumps(arguments, ensure_ascii=False)[:60]
            result_preview = result[:80] + "..." if len(result) > 80 else result
            print(f"    [{turn}] {tool_name}({args_str}) → {result_preview}")

        turn += 1

    metrics = workspace.get_metrics()
    metrics["input_tokens"] = total_input_tokens
    metrics["output_tokens"] = total_output_tokens
    metrics["total_tokens"] = total_input_tokens + total_output_tokens
    metrics["cost_usd"] = 0.0  # Local model
    return metrics


def run_evaluation(
    api_base: str = "http://localhost:8000/v1",
    model_name: str = "Qwen2.5-Coder-7B-Instruct",
    data_path: str = "data/prompts/eval_fixed.jsonl",
    horizon_path: str = "/root/autodl-tmp/horizon",
    max_samples: int = 999,
    max_turns: int = 50,
    temperature: float = 0.7,
    output_path: str = "experiments/results/qwen_baseline.json",
    run_id: int = 1,
):
    try:
        client = OpenAI(base_url=api_base, api_key="not-needed")

        # Test connection
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"vLLM server models: {available}")
        if available:
            model_name = available[0]  # Use whatever model is loaded

        # Load eval set
        items = []
        with open(data_path) as f:
            for line in f:
                items.append(json.loads(line))
                if len(items) >= max_samples:
                    break
        print(f"Loaded {len(items)} tasks")

        experiment_start(f"Qwen B0 run{run_id}", model_name, len(items))

        # Run episodes
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
                client=client,
                model_name=model_name,
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

            workspace.cleanup()

        # Summary
        total = len(episodes)
        resolved = sum(1 for e in episodes if e["resolved"])
        fully_resolved = sum(1 for e in episodes if e.get("fully_resolved"))
        first_try = sum(1 for e in episodes if e["first_try_valid"])
        fixed = sum(1 for e in episodes if e["fix_rate"])
        verify_passed = sum(1 for e in episodes if e.get("verify_passed"))

        print(f"\n{'='*70}")
        print(f"  Qwen Baseline — {model_name} (run {run_id})")
        print(f"{'='*70}")
        print(f"\n┌{'─'*50}┐")
        print(f"│ {'Total tasks':<30} {total:>18} │")
        print(f"│ {'API Resolved':<30} {f'{resolved}/{total} ({resolved*100//total}%)':>18} │")
        print(f"│ {'  ├ First-try Valid':<30} {f'{first_try}/{total} ({first_try*100//total}%)':>18} │")
        print(f"│ {'  └ Fixed after Error':<30} {f'{fixed}/{total} ({fixed*100//total}%)':>18} │")
        print(f"│ {'Verify Passed':<30} {f'{verify_passed}/{total} ({verify_passed*100//total}%)':>18} │")
        print(f"│ {'★ Fully Resolved':<30} {f'{fully_resolved}/{total} ({fully_resolved*100//total}%)':>18} │")
        avg_turns = sum(e["total_turns"] for e in episodes) / total
        avg_time = sum(e["wall_time"] for e in episodes) / total
        print(f"│ {'Avg turns':<30} {avg_turns:>18.1f} │")
        print(f"│ {'Avg time':<30} {f'{avg_time:.1f}s':>18} │")
        print(f"└{'─'*50}┘")

        # By level
        print(f"\n┌{'─'*82}┐")
        print(f"│ {'Level':<24} {'API OK':>8} {'Verify':>8} {'★Full':>8} {'1st✓':>8} {'Fix':>8} {'Turns':>8} │")
        print(f"├{'─'*82}┤")
        for lv in sorted(set(e["level"] for e in episodes)):
            lv_eps = [e for e in episodes if e["level"] == lv]
            n = len(lv_eps)
            lv_name = lv_eps[0].get("level_name", "")
            api_ok = sum(1 for e in lv_eps if e["resolved"])
            vfy_ok = sum(1 for e in lv_eps if e.get("verify_passed"))
            full_ok = sum(1 for e in lv_eps if e.get("fully_resolved"))
            fst_ok = sum(1 for e in lv_eps if e["first_try_valid"])
            fix_ok = sum(1 for e in lv_eps if e["fix_rate"])
            avg_t = sum(e["total_turns"] for e in lv_eps) / n
            print(f"│ L{lv} {lv_name:<21} {api_ok}/{n:>5} {vfy_ok}/{n:>5} {full_ok}/{n:>5} {fst_ok}/{n:>5} {fix_ok}/{n:>5} {avg_t:>8.1f} │")
        print(f"└{'─'*82}┘")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model_name,
            "experiment": f"qwen_baseline_run{run_id}",
            "total": total,
            "resolved_rate": resolved / total,
            "first_try_valid_rate": first_try / total,
            "fix_rate": fixed / total,
            "fully_resolved_rate": fully_resolved / total,
            "avg_turns": avg_turns,
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
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data-path", default="data/prompts/eval_fixed.jsonl")
    parser.add_argument("--horizon-path", default="/root/autodl-tmp/horizon")
    parser.add_argument("--max-samples", type=int, default=999)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-path", default="experiments/results/qwen_baseline.json")
    parser.add_argument("--run-id", type=int, default=1)
    args = parser.parse_args()

    run_evaluation(**vars(args))
