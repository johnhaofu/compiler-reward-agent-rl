"""
Claude API Agent Loop Baseline.

Uses Claude Sonnet as a strong baseline for comparison.
Same tools, same prompts, same validation — different model.
This establishes the "ceiling" that RL-trained models should approach.
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

# Convert our tool format to Anthropic format
def convert_tools_to_anthropic(tools: list) -> list:
    """Convert OpenAI-style tool defs to Anthropic format."""
    anthropic_tools = []
    for t in tools:
        func = t["function"]
        anthropic_tools.append({
            "name": func["name"],
            "description": func["description"],
            "input_schema": func["parameters"],
        })
    return anthropic_tools


ANTHROPIC_TOOLS = convert_tools_to_anthropic(TOOLS)


def run_claude_agent_episode(
    client: Anthropic,
    workspace: AgentWorkspace,
    task_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 20,
) -> dict:
    """
    Run one agent episode with Claude.

    Args:
        client: Anthropic client
        workspace: AgentWorkspace with tools
        task_prompt: User's task description
        model: Claude model name
        max_turns: Max tool call turns

    Returns:
        Episode metrics dict
    """
    messages = [{"role": "user", "content": task_prompt}]

    total_input_tokens = 0
    total_output_tokens = 0
    turn = 0

    while turn < max_turns and not workspace.is_done:
        # Call Claude
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=ANTHROPIC_TOOLS,
                messages=messages,
            )
        except Exception as e:
            print(f"    Claude API error: {e}")
            break

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Process response
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Check if there are tool calls
        tool_uses = [b for b in assistant_content if b.type == "tool_use"]

        if not tool_uses:
            # No tool calls — Claude finished with text
            break

        # Execute tool calls
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

            # Print progress
            result_preview = result[:100] + "..." if len(result) > 100 else result
            print(f"    Turn {turn}: {tool_use.name}({json.dumps(tool_use.input)[:60]}) → {result_preview}")

        messages.append({"role": "user", "content": tool_results})
        turn += 1

    # Collect metrics
    metrics = workspace.get_metrics()
    metrics["model"] = model
    metrics["input_tokens"] = total_input_tokens
    metrics["output_tokens"] = total_output_tokens
    metrics["total_tokens"] = total_input_tokens + total_output_tokens

    # Estimate cost (Sonnet 4 pricing)
    pricing = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.0},
    }
    model_pricing = pricing.get(model, pricing["claude-sonnet-4-20250514"])
    metrics["cost_usd"] = (
        total_input_tokens * model_pricing["input"] / 1_000_000
        + total_output_tokens * model_pricing["output"] / 1_000_000
    )

    return metrics


def run_evaluation(
    data_path: str = "data/prompts/val.jsonl",
    horizon_path: str = "/tmp/horizon",
    max_samples: int = 50,
    max_turns: int = 20,
    model: str = "claude-sonnet-4-20250514",
    output_path: str = "experiments/results/claude_baseline.json",
):
    """Run Claude agent loop evaluation."""
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Try reading from config
            config_paths = [
                Path.home() / ".anthropic" / "api_key",
                Path.home() / ".config" / "anthropic" / "api_key",
            ]
            for cp in config_paths:
                if cp.exists():
                    api_key = cp.read_text().strip()
                    break

        if not api_key:
            print("ERROR: Set ANTHROPIC_API_KEY environment variable")
            return

        client = Anthropic(api_key=api_key)

        # Load prompts
        prompts_data = []
        with open(data_path) as f:
            for line in f:
                prompts_data.append(json.loads(line))
                if len(prompts_data) >= max_samples:
                    break
        print(f"Loaded {len(prompts_data)} prompts")

        experiment_start(f"Claude Baseline ({model.split('-')[1]})", model, len(prompts_data))

        # Run episodes
        episodes = []
        total_cost = 0

        for i, item in enumerate(prompts_data):
            task_prompt = item["prompt"][-1]["content"]
            tpl_type = item["template_type"]

            # Fresh workspace per episode
            workspace = AgentWorkspace(
                horizon_path=horizon_path,
                components_path="data/horizon_components.json",
                schemas_dir="data/schemas",
            )

            print(f"\n[{i+1}/{len(prompts_data)}] {tpl_type} ({item.get('industry', '')})")

            t0 = time.time()
            metrics = run_claude_agent_episode(
                client=client,
                workspace=workspace,
                task_prompt=task_prompt,
                model=model,
                max_turns=max_turns,
            )
            metrics["wall_time"] = round(time.time() - t0, 2)
            metrics["template_type"] = tpl_type
            metrics["industry"] = item.get("industry", "")
            metrics["complexity"] = item.get("complexity", "")

            episodes.append(metrics)
            total_cost += metrics.get("cost_usd", 0)

            status = "✅" if metrics["pass_at_final"] else "❌"
            print(f"  {status} turns={metrics['total_turns']} "
                  f"validates={metrics['validate_calls']} "
                  f"pass@1={metrics['pass_at_1']} "
                  f"tokens={metrics['total_tokens']} "
                  f"${metrics['cost_usd']:.4f}")

            workspace.cleanup()

        # Summary
        total = len(episodes)
        pass_at_1 = sum(1 for e in episodes if e["pass_at_1"])
        pass_final = sum(1 for e in episodes if e["pass_at_final"])
        avg_turns = sum(e["total_turns"] for e in episodes) / total
        avg_validates = sum(e["validate_calls"] for e in episodes) / total

        print(f"\n{'='*60}")
        print(f"Claude Baseline Results ({model})")
        print(f"{'='*60}")
        print(f"Total tasks:      {total}")
        print(f"Pass@1:           {pass_at_1}/{total} ({pass_at_1*100//total}%)")
        print(f"Pass@final:       {pass_final}/{total} ({pass_final*100//total}%)")
        print(f"Avg turns:        {avg_turns:.1f}")
        print(f"Avg validates:    {avg_validates:.1f}")
        print(f"Total cost:       ${total_cost:.4f}")
        print(f"Cost per task:    ${total_cost/total:.4f}")

        # By complexity
        for comp in ["low", "medium", "high"]:
            comp_eps = [e for e in episodes if e["complexity"] == comp]
            if comp_eps:
                cp1 = sum(1 for e in comp_eps if e["pass_at_1"])
                cpf = sum(1 for e in comp_eps if e["pass_at_final"])
                print(f"  {comp}: pass@1={cp1}/{len(comp_eps)} pass@final={cpf}/{len(comp_eps)}")

        # Tool usage analysis
        all_sequences = [e["tool_sequence"] for e in episodes]
        tool_counts = {}
        for seq in all_sequences:
            for t in seq:
                tool_counts[t] = tool_counts.get(t, 0) + 1
        print(f"\nTool usage:")
        for t, c in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c} ({c/total:.1f}/episode)")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model,
            "experiment": "claude_baseline",
            "total": total,
            "pass_at_1": pass_at_1 / total,
            "pass_at_final": pass_final / total,
            "avg_turns": avg_turns,
            "avg_validates": avg_validates,
            "total_cost_usd": total_cost,
            "episodes": episodes,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

        experiment_done(f"Claude Baseline ({model.split('-')[1]})", summary)

    except Exception as e:
        experiment_error("Claude Baseline", str(e))
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/prompts/val.jsonl")
    parser.add_argument("--horizon-path", type=str, default="/tmp/horizon")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--output", type=str, default="experiments/results/claude_baseline.json")
    args = parser.parse_args()

    run_evaluation(
        data_path=args.data_path,
        horizon_path=args.horizon_path,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        model=args.model,
        output_path=args.output,
    )
