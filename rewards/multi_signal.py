"""
Multi-Signal Reward: 4 binary rewards combined.
json_valid + sections_valid + theme_check_valid + schema_valid
"""


def multi_signal_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Sum of 4 binary rewards (0-4 range, then normalized to 0-1).

    Each check independently contributes:
      json_valid:        +1 if JSON parses correctly
      sections_valid:    +1 if all referenced sections exist
      theme_check_valid: +1 if shopify theme check passes
      all_passed:        +1 bonus if everything passes
    """
    env = kwargs["env"]
    template_types = kwargs["template_types"]
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for response, tpl_type in zip(responses, template_types):
        result = env.validate(tpl_type, response)
        rd = result.to_reward_dict()
        # Sum of binary signals, normalized to [0, 1]
        score = (rd["json_valid"] + rd["sections_valid"] + rd["theme_check_valid"]) / 3.0
        # Bonus for all passing
        if result.all_passed:
            score = 1.0
        rewards.append(score)

    return rewards


def json_reward(prompts, completions, **kwargs) -> list[float]:
    """Binary: JSON parses correctly."""
    env = kwargs["env"]
    template_types = kwargs["template_types"]
    responses = [c[0]["content"] for c in completions]
    return [
        1.0 if env.validate(t, r).json_valid else 0.0
        for r, t in zip(responses, template_types)
    ]


def sections_reward(prompts, completions, **kwargs) -> list[float]:
    """Binary: All referenced sections/blocks exist."""
    env = kwargs["env"]
    template_types = kwargs["template_types"]
    responses = [c[0]["content"] for c in completions]
    return [
        1.0 if env.validate(t, r).sections_valid else 0.0
        for r, t in zip(responses, template_types)
    ]


def theme_check_reward(prompts, completions, **kwargs) -> list[float]:
    """Binary: shopify theme check passes."""
    env = kwargs["env"]
    template_types = kwargs["template_types"]
    responses = [c[0]["content"] for c in completions]
    return [
        1.0 if env.validate(t, r).theme_check_valid else 0.0
        for r, t in zip(responses, template_types)
    ]
