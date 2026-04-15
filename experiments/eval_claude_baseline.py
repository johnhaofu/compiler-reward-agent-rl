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
from evaluation.verify_engine import verify_episode
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

            # Run verify checks (Layer 2: requirements verification)
            verify_result = verify_episode(item, metrics)
            metrics["verify_passed"] = verify_result["passed"]
            metrics["verify_total"] = verify_result["total_checks"]
            metrics["verify_passed_count"] = verify_result["passed_checks"]
            metrics["verify_failed"] = verify_result["failed_checks"]

            # Fully resolved = API passed + verify passed
            metrics["fully_resolved"] = metrics["resolved"] and verify_result["passed"]

            episodes.append(metrics)
            total_cost += metrics.get("cost_usd", 0)

            api_status = "✅" if metrics["resolved"] else "❌"
            verify_status = "✓" if verify_result["passed"] else f"✗{verify_result['passed_checks']}/{verify_result['total_checks']}"
            ftv = "1st✓" if metrics["first_try_valid"] else "fix" if metrics["fix_rate"] else "fail"
            print(f"  {api_status} API [{ftv}] | verify:{verify_status} | "
                  f"turns={metrics['total_turns']} ${metrics['cost_usd']:.3f}")
            if verify_result["failed_checks"]:
                for fc in verify_result["failed_checks"][:2]:
                    print(f"    verify fail: {fc['reason'][:80]}")

            workspace.cleanup()

        # ═══ Summary ═══
        total = len(episodes)
        resolved = sum(1 for e in episodes if e["resolved"])
        fully_resolved = sum(1 for e in episodes if e.get("fully_resolved", False))
        first_try = sum(1 for e in episodes if e["first_try_valid"])
        fixed = sum(1 for e in episodes if e["fix_rate"])
        verify_passed = sum(1 for e in episodes if e.get("verify_passed", False))
        failed = total - fully_resolved
        resolved_eps = [e for e in episodes if e["resolved"]]

        print(f"\n{'='*70}")
        print(f"  Results — {model}")
        print(f"{'='*70}")

        # Core metrics table
        print(f"\n┌{'─'*50}┐")
        print(f"│ {'Metric':<30} {'Value':>18} │")
        print(f"├{'─'*50}┤")
        print(f"│ {'Total tasks':<30} {total:>18} │")
        print(f"│ {'API Resolved':<30} {f'{resolved}/{total} ({resolved*100//total}%)':>18} │")
        print(f"│ {'  ├ First-try Valid':<30} {f'{first_try}/{total} ({first_try*100//total}%)':>18} │")
        print(f"│ {'  └ Fixed after Error':<30} {f'{fixed}/{total} ({fixed*100//total}%)':>18} │")
        print(f"│ {'Verify Passed':<30} {f'{verify_passed}/{total} ({verify_passed*100//total}%)':>18} │")
        print(f"│ {'★ Fully Resolved':<30} {f'{fully_resolved}/{total} ({fully_resolved*100//total}%)':>18} │")
        print(f"│ {'Failed':<30} {f'{failed}/{total} ({failed*100//total}%)':>18} │")
        print(f"├{'─'*50}┤")
        avg_turns = sum(e["total_turns"] for e in episodes) / total
        avg_validates = sum(e["validate_attempts"] for e in episodes) / total
        avg_resolve_turns = sum(e["turns_to_resolve"] for e in resolved_eps) / max(len(resolved_eps), 1) if resolved_eps else 0
        avg_fix_turns = sum(e["fix_turns"] for e in episodes if e["fix_rate"]) / max(fixed, 1) if fixed else 0
        print(f"│ {'Avg turns / task':<30} {avg_turns:>18.1f} │")
        print(f"│ {'Avg validates / task':<30} {avg_validates:>18.1f} │")
        print(f"│ {'Avg turns to resolve':<30} {avg_resolve_turns:>18.1f} │")
        print(f"│ {'Avg fix turns (fix cases)':<30} {avg_fix_turns:>18.1f} │")
        print(f"├{'─'*50}┤")
        print(f"│ {'Total cost':<30} {f'${total_cost:.3f}':>18} │")
        cost_per_task = total_cost / total
        cost_per_resolve = total_cost / max(resolved, 1)
        print(f"│ {'Cost / task':<30} {f'${cost_per_task:.3f}':>18} │")
        print(f"│ {'Cost / resolved task':<30} {f'${cost_per_resolve:.3f}':>18} │")
        print(f"└{'─'*50}┘")

        # By level
        print(f"\n┌{'─'*82}┐")
        print(f"│ {'Level':<24} {'API OK':>8} {'Verify':>8} {'★Full':>8} {'1st✓':>8} {'Fix':>8} {'Turns':>8} │")
        print(f"├{'─'*82}┤")
        for lv in sorted(set(e["level"] for e in episodes)):
            lv_eps = [e for e in episodes if e["level"] == lv]
            lv_name = lv_eps[0].get("level_name", "")
            n = len(lv_eps)
            lv_resolved = sum(1 for e in lv_eps if e["resolved"])
            lv_verify = sum(1 for e in lv_eps if e.get("verify_passed", False))
            lv_full = sum(1 for e in lv_eps if e.get("fully_resolved", False))
            lv_first = sum(1 for e in lv_eps if e["first_try_valid"])
            lv_fixed = sum(1 for e in lv_eps if e["fix_rate"])
            lv_turns = sum(e["total_turns"] for e in lv_eps) / n
            print(f"│ L{lv} {lv_name:<21} {f'{lv_resolved}/{n}':>8} {f'{lv_verify}/{n}':>8} {f'{lv_full}/{n}':>8} {f'{lv_first}/{n}':>8} {f'{lv_fixed}/{n}':>8} {lv_turns:>8.1f} │")
        print(f"└{'─'*82}┘")

        # Error type analysis
        all_error_types = {}
        for e in episodes:
            for etype, cnt in e.get("error_types", {}).items():
                all_error_types[etype] = all_error_types.get(etype, 0) + cnt
        if all_error_types:
            print(f"\nError types:")
            for etype, cnt in sorted(all_error_types.items(), key=lambda x: -x[1]):
                print(f"  {etype:<30} {cnt:>5}")

        # Tool usage
        all_tool_counts = {}
        for e in episodes:
            for t, c in e.get("tool_counts", {}).items():
                all_tool_counts[t] = all_tool_counts.get(t, 0) + c
        print(f"\nTool usage (total / per-task):")
        for t, c in sorted(all_tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:<25} {c:>5} ({c/total:.1f}/task)")

        # Unresolved
        unresolved_eps = [e for e in episodes if not e["resolved"]]
        if unresolved_eps:
            print(f"\nUnresolved tasks ({len(unresolved_eps)}):")
            for e in unresolved_eps:
                errs = e.get("unique_errors", [])
                last_err = errs[0][:80] if errs else "no error"
                print(f"  {e['task_id']} L{e['level']} {e['template_type']}: {last_err}")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        def level_stats(lv):
            lv_eps = [e for e in episodes if e["level"] == lv]
            n = max(len(lv_eps), 1)
            return {
                "total": len(lv_eps),
                "resolved_rate": sum(1 for e in lv_eps if e["resolved"]) / n,
                "first_try_valid_rate": sum(1 for e in lv_eps if e["first_try_valid"]) / n,
                "fix_rate": sum(1 for e in lv_eps if e["fix_rate"]) / n,
                "avg_turns": sum(e["total_turns"] for e in lv_eps) / n,
                "avg_validates": sum(e["validate_attempts"] for e in lv_eps) / n,
            }

        summary = {
            "model": model,
            "experiment": "claude_baseline",
            "total": total,
            "resolved_rate": resolved / total,
            "first_try_valid_rate": first_try / total,
            "fix_rate": fixed / total,
            "avg_turns": avg_turns,
            "avg_validates": avg_validates,
            "avg_turns_to_resolve": avg_resolve_turns,
            "total_cost_usd": total_cost,
            "cost_per_task": cost_per_task,
            "cost_per_resolve": cost_per_resolve,
            "error_types": all_error_types,
            "by_level": {
                lv: level_stats(lv)
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
    parser.add_argument("--output-path", default="experiments/results/claude_baseline.json")
    args = parser.parse_args()

    run_evaluation(
        data_path=args.data_path,
        horizon_path=args.horizon_path,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        model=args.model,
        output_path=args.output_path,
    )
