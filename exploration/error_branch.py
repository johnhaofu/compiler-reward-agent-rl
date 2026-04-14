"""
Error-Triggered Branching.

Core innovation: Use compilation failure as a natural branching signal
for adaptive exploration, with zero computational overhead.

Instead of computing token entropy (ARPO) to decide when to branch,
we simply branch when the compiler reports an error — a free,
deterministic signal that naturally correlates with high uncertainty.
"""

from dataclasses import dataclass


@dataclass
class Branch:
    """A branched generation from an error point."""
    branch_id: int
    strategy: str          # "retry" | "hint_retry" | "partial_fix"
    prompt: str            # The prompt for this branch
    response: str = ""     # Generated response
    reward: float = 0.0    # Reward for this branch


def create_branches(
    original_prompt: str,
    error_message: str,
    num_branches: int = 3,
) -> list[Branch]:
    """
    Create branched generation prompts from a compilation error.

    Three branching strategies:
      1. retry: Same prompt, different sampling (temperature diversity)
      2. hint_retry: Append error message as hint
      3. partial_fix: Ask to fix only the erroring part

    Args:
        original_prompt: The original generation prompt
        error_message: Compiler/validator error message
        num_branches: Number of branches to create

    Returns:
        List of Branch objects with prompts ready for generation
    """
    branches = []

    # Strategy 1: Retry with same prompt (rely on sampling diversity)
    branches.append(Branch(
        branch_id=0,
        strategy="retry",
        prompt=original_prompt,
    ))

    # Strategy 2: Append error as hint
    if num_branches >= 2:
        branches.append(Branch(
            branch_id=1,
            strategy="hint_retry",
            prompt=f"{original_prompt}\n\n[Previous attempt had this error, please fix it]\n{error_message}",
        ))

    # Strategy 3: Partial fix — only regenerate the problematic section
    if num_branches >= 3:
        branches.append(Branch(
            branch_id=2,
            strategy="partial_fix",
            prompt=f"{original_prompt}\n\n[Fix ONLY the following error, keep everything else]\n{error_message}",
        ))

    return branches[:num_branches]


def error_branch_exploration(
    model,
    tokenizer,
    prompts: list[str],
    template_types: list[str],
    env,
    num_branches: int = 3,
    sampling_params=None,
) -> dict:
    """
    Generate with error-triggered branching.

    Flow:
      1. Generate initial response for each prompt
      2. Validate each response
      3. For failed responses: create branches and generate alternatives
      4. Return all responses with their rewards

    Args:
        model: The language model (with generate method)
        tokenizer: The tokenizer
        prompts: List of generation prompts
        template_types: Corresponding template types
        env: HorizonEnvironment
        num_branches: Number of branches per error
        sampling_params: vLLM sampling params

    Returns:
        Dict with responses, rewards, branch_info
    """
    all_responses = []
    all_rewards = []
    all_branch_info = []

    # Step 1: Initial generation
    # (Actual generation delegated to the trainer — this is the logic layer)

    for prompt, tpl_type in zip(prompts, template_types):
        # Placeholder: in practice, model.generate() is called by the trainer
        # Here we define the branching logic that wraps around generation

        branch_info = {
            "prompt": prompt,
            "template_type": tpl_type,
            "branches": [],
            "error_triggered": False,
        }
        all_branch_info.append(branch_info)

    return {
        "branch_info": all_branch_info,
        "num_branches": num_branches,
    }


def compute_branch_advantages(
    original_reward: float,
    branch_rewards: list[float],
    shared_prefix_len: int,
    total_lens: list[int],
) -> dict:
    """
    Compute advantages with prefix/suffix separation (ARPO-style).

    Shared prefix tokens: average advantage across all branches
    Independent suffix tokens: each branch gets its own advantage

    Args:
        original_reward: Reward for the original (unbranched) response
        branch_rewards: Rewards for each branch
        shared_prefix_len: Number of tokens in the shared prefix
        total_lens: Total length of each response

    Returns:
        Dict with prefix_advantage and per-branch suffix_advantages
    """
    all_rewards = [original_reward] + branch_rewards
    mean_r = sum(all_rewards) / len(all_rewards)
    std_r = max((sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5, 1e-8)

    # Prefix advantage: average across all branches
    prefix_advantages = [(r - mean_r) / std_r for r in all_rewards]
    prefix_avg = sum(prefix_advantages) / len(prefix_advantages)

    # Suffix advantages: individual per branch
    suffix_advantages = [(r - mean_r) / std_r for r in all_rewards]

    return {
        "prefix_advantage": prefix_avg,
        "suffix_advantages": suffix_advantages,
        "shared_prefix_len": shared_prefix_len,
    }
