"""
Claude API Agent Loop Baseline.

Uses Claude as a strong baseline with full tool access.
Same tools, same prompts, same validation — different model.
Supports 3-level eval: assemble page / generate component / full.
"""

import json
import sys
import time
import os
from pathlib import Path
from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.tools import TOOLS, SYSTEM_PROMPT, AgentWorkspace
from experiments.notify import experiment_start, experiment_done, experiment_error


def convert_tools_to_anthropic(tools: list) -> list:
    """Convert OpenAI-style tool defs to Anthropic format."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]


ANTHROPIC_TOOLS = convert_tools_to_anthropic(TOOLS)


def run_episode(
    client: Anthropic,
    workspace: AgentWorkspace,
    task_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 20,
) -> dict:
    """Run one agent episode with Claude."""
    messages = [{"role": "user", "content": task_prompt}]

    total_input_tokens = 0
    total_output_tokens = 0
    turn = 0

    while turn < max_turns and not workspace.is_done:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=ANTHROPIC_TOOLS,
                messages=messages,
            )
        except Exception as e:
            print(f"    API error: {e}")
            break

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Extract tool calls
        tool_uses = [b for b in assistant_content if b.type == "tool_use"]

        if not tool_uses:
            # Text-only response, no tool calls
            break

        # Execute tools
        tool_results = []
        for tool_use in tool_uses:
            result = workspace.execute_tool(
                turn=turn,
                tool_name=tool_use.name,
                arguments=tool_use.input,
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

            # Progress
            args_str = json.dumps(tool_use.input, ensure_ascii=False)[:60]
            result_preview = result[:80] + "..." if len(result) > 80 else result
            print(f"    [{turn}] {tool_use.name}({args_str}) → {result_preview}")

        messages.append({"role": "user", "content": tool_results})
        turn += 1

    # Metrics
    metrics = workspace.get_metrics()
    metrics["input_tokens"] = total_input_tokens
    metrics["output_tokens"] = total_output_tokens
    metrics["total_tokens"] = total_input_tokens + total_output_tokens

    # Cost estimation
    pricing = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.0},
    }
    p = pricing.get(model, pricing["claude-sonnet-4-20250514"])
    metrics["cost_usd"] = (
        total_input_tokens * p["input"] / 1_000_000
        + total_output_tokens * p["output"] / 1_000_000
    )
    return metrics


def run_evaluation(
    data_path: str = "data/prompts/eval_fixed.jsonl",
    horizon_path: str = "/tmp/horizon",
    max_samples: int = 999,
    max_turns: int = 20,
    model: str = "claude-sonnet-4-20250514",
    output_path: str = "experiments/results/claude_baseline.json",
):
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: Set ANTHROPIC_API_KEY")
            return
        client = Anthropic(api_key=api_key)

        # Load eval set
        items = []
        with open(data_path) as f:
            for line in f:
                items.append(json.loads(line))
                if len(items) >= max_samples:
                    break

        print(f"Loaded {len(items)} tasks from {data_path}")
        level_counts = {}
        for item in items:
            lv = item.get("level", 0)
            level_counts[lv] = level_counts.get(lv, 0) + 1
        for lv, cnt in sorted(level_counts.items()):
            print(f"  Level {lv}: {cnt} tasks")

        experiment_start(f"Claude {model.split('-')[1]}", model, len(items))

        # Run all episodes
        episodes = []
        total_cost = 0

        for i, item in enumerate(items):
            task_id = item.get("id", f"task-{i}")
            level = item.get("level", 0)
            level_name = item.get("level_name", "unknown")
            tpl_type = item.get("template_type", "")
            industry = item.get("industry", "")
            task_prompt = item["prompt"][-1]["content"]

            workspace = AgentWorkspace(
                horizon_path=horizon_path,
                components_path="data/horizon_components.json",
                schemas_dir="data/schemas",
            )

            print(f"\n[{i+1}/{len(items)}] {task_id} | L{level} {level_name} | {tpl_type} ({industry})")

            t0 = time.time()
            metrics = run_episode(
                client=client,
                workspace=workspace,
                task_prompt=task_prompt,
                model=model,
                max_turns=max_turns,
            )
            metrics["wall_time"] = round(time.time() - t0, 2)
            metrics["task_id"] = task_id
            metrics["level"] = level
            metrics["level_name"] = level_name
            metrics["template_type"] = tpl_type
            metrics["industry"] = industry

            episodes.append(metrics)
            total_cost += metrics.get("cost_usd", 0)

            status = "✅" if metrics["pass_at_final"] else "❌"
            print(f"  {status} turns={metrics['total_turns']} "
                  f"validates={metrics['validate_calls']} "
                  f"pass@1={'Y' if metrics['pass_at_1'] else 'N'} "
                  f"${metrics['cost_usd']:.3f}")

            workspace.cleanup()

        # ═══ Summary ═══
        total = len(episodes)
        pass_1 = sum(1 for e in episodes if e["pass_at_1"])
        pass_f = sum(1 for e in episodes if e["pass_at_final"])

        print(f"\n{'='*70}")
        print(f"Claude Baseline Results — {model}")
        print(f"{'='*70}")
        print(f"{'Metric':<25} {'Value':>10}")
        print(f"{'-'*40}")
        print(f"{'Total tasks':<25} {total:>10}")
        print(f"{'Pass@1':<25} {f'{pass_1}/{total} ({pass_1*100//total}%)':>10}")
        print(f"{'Pass@final':<25} {f'{pass_f}/{total} ({pass_f*100//total}%)':>10}")
        print(f"{'Avg turns':<25} {sum(e['total_turns'] for e in episodes)/total:>10.1f}")
        print(f"{'Avg validates':<25} {sum(e['validate_calls'] for e in episodes)/total:>10.1f}")
        print(f"{'Total cost':<25} {f'${total_cost:.3f}':>10}")

        # By level
        print(f"\n{'Level':<30} {'Pass@1':>10} {'Pass@final':>12} {'Avg turns':>10}")
        print(f"{'-'*65}")
        for lv in sorted(set(e["level"] for e in episodes)):
            lv_eps = [e for e in episodes if e["level"] == lv]
            lv_name = lv_eps[0].get("level_name", "")
            lv_p1 = sum(1 for e in lv_eps if e["pass_at_1"])
            lv_pf = sum(1 for e in lv_eps if e["pass_at_final"])
            lv_turns = sum(e["total_turns"] for e in lv_eps) / len(lv_eps)
            print(f"L{lv} {lv_name:<27} {f'{lv_p1}/{len(lv_eps)}':>10} {f'{lv_pf}/{len(lv_eps)}':>12} {lv_turns:>10.1f}")

        # Tool usage
        tool_counts = {}
        for e in episodes:
            for t in e.get("tool_sequence", []):
                tool_counts[t] = tool_counts.get(t, 0) + 1
        print(f"\nTool usage (total / per-episode):")
        for t, c in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:<25} {c:>5} ({c/total:.1f}/ep)")

        # Unresolved errors
        unresolved = [e for e in episodes if not e["pass_at_final"]]
        if unresolved:
            print(f"\nUnresolved tasks ({len(unresolved)}):")
            for e in unresolved:
                errs = e.get("errors_encountered", [])
                last_err = errs[-1][:80] if errs else "no error captured"
                print(f"  {e['task_id']} L{e['level']} {e['template_type']}: {last_err}")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model,
            "experiment": "claude_baseline",
            "total": total,
            "pass_at_1": pass_1 / total,
            "pass_at_final": pass_f / total,
            "avg_turns": sum(e["total_turns"] for e in episodes) / total,
            "total_cost_usd": total_cost,
            "by_level": {
                lv: {
                    "total": len([e for e in episodes if e["level"] == lv]),
                    "pass_at_1": sum(1 for e in episodes if e["level"] == lv and e["pass_at_1"]) / max(len([e for e in episodes if e["level"] == lv]), 1),
                    "pass_at_final": sum(1 for e in episodes if e["level"] == lv and e["pass_at_final"]) / max(len([e for e in episodes if e["level"] == lv]), 1),
                }
                for lv in sorted(set(e["level"] for e in episodes))
            },
            "episodes": episodes,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

        experiment_done(f"Claude {model.split('-')[1]}", summary)

    except Exception as e:
        experiment_error("Claude Baseline", str(e))
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/prompts/eval_fixed.jsonl")
    parser.add_argument("--horizon-path", default="/tmp/horizon")
    parser.add_argument("--max-samples", type=int, default=999)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--output", default="experiments/results/claude_baseline.json")
    args = parser.parse_args()

    run_evaluation(**vars(args))
