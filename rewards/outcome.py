"""
Outcome Reward: Binary compile pass/fail.
Baseline reward function — only checks final result.
"""


def outcome_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Binary reward: 1.0 if all validation passes, 0.0 otherwise.

    Args:
        prompts: List of prompt dicts
        completions: List of completion dicts from TRL
        **kwargs: Must include 'env' (HorizonEnvironment) and 'template_types'
    """
    env = kwargs["env"]
    template_types = kwargs["template_types"]
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for response, tpl_type in zip(responses, template_types):
        result = env.validate(tpl_type, response)
        rewards.append(1.0 if result.all_passed else 0.0)

    return rewards
