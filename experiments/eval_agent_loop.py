"""
Agent Loop Evaluation.

Real evaluation: generate → validate → feed error back → fix → validate → ...
until pass or max_turns reached.

This is the core evaluation that shows the value of:
- Compiler feedback as reward signal (errors guide the fix)
- Error-triggered branching (fork multiple fix strategies on failure)
- Process reward (intermediate feedback at each turn)
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.sitemuse_validator import SitemuseValidator, ValidationResult
from experiments.notify import experiment_start, experiment_done, experiment_error


@dataclass
class AgentTurn:
    """One turn of the agent loop."""
    turn: int
    prompt: str
    response: str
    validation: dict
    passed: bool
    error_message: str = ""
    gen_time: float = 0.0


@dataclass
class AgentEpisode:
    """A complete agent episode (all turns for one task)."""
    template_type: str
    industry: str
    complexity: str
    turns: list[AgentTurn] = field(default_factory=list)
    passed: bool = False
    total_turns: int = 0
    total_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "template_type": self.template_type,
            "industry": self.industry,
            "complexity": self.complexity,
            "passed": self.passed,
            "total_turns": self.total_turns,
            "total_time": round(self.total_time, 2),
            "turns": [
                {
                    "turn": t.turn,
                    "passed": t.passed,
                    "error": t.error_message,
                    "response_len": len(t.response),
                    "gen_time": round(t.gen_time, 2),
                }
                for t in self.turns
            ],
        }


SYSTEM_PROMPT = """You are an expert Shopify theme developer specializing in the Horizon theme.
Generate valid Shopify template JSON files that are compatible with the Horizon theme.

Rules:
- Output ONLY valid JSON (no comments, no explanation, no markdown code blocks)
- The JSON must have "sections" and "order" keys
- Only use section types that exist in the Horizon theme
- Only use block types that exist in the Horizon theme
- Include appropriate settings for each section and block"""

FIX_PROMPT_TEMPLATE = """The previous template had the following error:

{error}

Please fix the template and output the corrected JSON. Output ONLY valid JSON, no explanation."""


def run_agent_loop(
    model,
    tokenizer,
    validator: SitemuseValidator,
    task_prompt: str,
    template_type: str,
    max_turns: int = 5,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> AgentEpisode:
    """
    Run one agent episode: generate → validate → fix → validate → ...

    Args:
        model: Model with fast_generate method
        tokenizer: Tokenizer
        validator: SitemuseValidator
        task_prompt: The user's task description
        template_type: e.g. "page.about"
        max_turns: Max fix attempts
        max_new_tokens: Max tokens per generation
        temperature: Sampling temperature

    Returns:
        AgentEpisode with all turns and final result
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
    )

    episode = AgentEpisode(
        template_type=template_type,
        industry="",
        complexity="",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    last_response = ""

    for turn_idx in range(max_turns):
        # Build prompt
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        t0 = time.time()
        outputs = model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )
        gen_time = time.time() - t0

        response = outputs[0].outputs[0].text
        last_response = response

        # Validate
        val_result = validator.validate(template_type, response)

        turn = AgentTurn(
            turn=turn_idx,
            prompt=messages[-1]["content"][:200],
            response=response,
            validation=val_result.to_reward_dict(),
            passed=val_result.all_passed,
            error_message=val_result.get_error_message(),
            gen_time=gen_time,
        )
        episode.turns.append(turn)

        if val_result.all_passed:
            episode.passed = True
            break

        # Not passed → add error feedback and ask for fix
        error_msg = val_result.get_error_message()
        if not error_msg:
            # No specific error to feed back, generic retry
            error_msg = "The template is invalid. Please try again."

        # Append assistant response and error feedback for next turn
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": FIX_PROMPT_TEMPLATE.format(error=error_msg),
        })

    episode.total_turns = len(episode.turns)
    episode.total_time = sum(t.gen_time for t in episode.turns)
    return episode


def run_evaluation(
    model_path: str,
    data_path: str = "data/prompts/val.jsonl",
    max_samples: int = 50,
    max_turns: int = 5,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    output_path: str = "experiments/results/agent_loop.json",
):
    """Run full agent loop evaluation."""
    try:
        # 1. Load model
        print("Loading model with vLLM...")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=8192,
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
        print(f"Loaded {len(prompts_data)} prompts")

        # 3. Init validator
        validator = SitemuseValidator()

        experiment_start(
            f"Agent Loop (max_turns={max_turns})",
            model_path.split("/")[-1],
            len(prompts_data),
        )

        # 4. Run agent loop for each prompt
        episodes = []

        for i, item in enumerate(prompts_data):
            task_prompt = item["prompt"][-1]["content"]  # user message
            tpl_type = item["template_type"]

            episode = run_agent_loop(
                model=model,
                tokenizer=tokenizer,
                validator=validator,
                task_prompt=task_prompt,
                template_type=tpl_type,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            episode.industry = item.get("industry", "")
            episode.complexity = item.get("complexity", "")
            episodes.append(episode)

            status = "✅" if episode.passed else "❌"
            turns_info = f"turn {episode.total_turns}/{max_turns}"
            print(f"  [{i+1}/{len(prompts_data)}] {status} {tpl_type:20s} | "
                  f"{turns_info} | {episode.total_time:.1f}s"
                  + (f" | last error: {episode.turns[-1].error_message[:60]}" if not episode.passed else ""))

        # 5. Summary
        total = len(episodes)
        passed = sum(1 for e in episodes if e.passed)
        avg_turns = sum(e.total_turns for e in episodes) / total

        # Pass@1: passed on first try
        pass_at_1 = sum(1 for e in episodes if e.turns[0].passed)

        # Pass by turn
        pass_by_turn = {}
        for t in range(max_turns):
            passed_by_t = sum(
                1 for e in episodes
                if any(turn.passed for turn in e.turns[:t+1])
            )
            pass_by_turn[f"pass@turn{t+1}"] = passed_by_t / total

        print(f"\n{'='*60}")
        print(f"Agent Loop Results ({model_path.split('/')[-1]})")
        print(f"{'='*60}")
        print(f"Total tasks:      {total}")
        print(f"Pass@1 (no fix):  {pass_at_1}/{total} ({pass_at_1*100//total}%)")
        print(f"Pass@{max_turns} (with fix): {passed}/{total} ({passed*100//total}%)")
        print(f"Avg turns used:   {avg_turns:.1f}")
        print(f"Avg time/task:    {sum(e.total_time for e in episodes)/total:.1f}s")

        print(f"\nPass rate by turn:")
        for k, v in pass_by_turn.items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"  {k}: {bar} {v*100:.1f}%")

        # By complexity
        print(f"\nBy complexity:")
        for comp in ["low", "medium", "high"]:
            comp_eps = [e for e in episodes if e.complexity == comp]
            if comp_eps:
                cp = sum(1 for e in comp_eps if e.passed)
                cp1 = sum(1 for e in comp_eps if e.turns[0].passed)
                print(f"  {comp}: pass@1={cp1}/{len(comp_eps)} "
                      f"pass@{max_turns}={cp}/{len(comp_eps)}")

        # Error analysis
        final_errors = [e.turns[-1].error_message for e in episodes if not e.passed]
        if final_errors:
            print(f"\nTop unresolved errors:")
            from collections import Counter
            for err, cnt in Counter(final_errors).most_common(5):
                print(f"  {cnt}x: {err[:100]}")

        # 6. Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model_path,
            "experiment": f"agent_loop_turns{max_turns}",
            "total": total,
            "pass_at_1": pass_at_1 / total,
            "pass_at_max": passed / total,
            "avg_turns": avg_turns,
            "max_turns": max_turns,
            "pass_by_turn": pass_by_turn,
            "episodes": [e.to_dict() for e in episodes],
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

        experiment_done(f"Agent Loop (turns={max_turns})", summary)

    except Exception as e:
        experiment_error("Agent Loop", str(e))
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/root/autodl-tmp/models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--data-path", type=str, default="data/prompts/val.jsonl")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="experiments/results/agent_loop.json")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        data_path=args.data_path,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        output_path=args.output,
    )
