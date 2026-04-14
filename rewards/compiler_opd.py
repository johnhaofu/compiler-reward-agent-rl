"""
Compiler-Guided OPD (On-Policy Distillation).

Core innovation: Use compiler error messages directly as hints for
constructing teacher distributions, without needing an LLM judge.

Compiler errors like "undefined variable 'titel' on line 4" are:
  - More precise than LLM-extracted hints
  - Deterministic (same input → same output)
  - Zero cost (no additional LLM calls)
"""

from dataclasses import dataclass


@dataclass
class OPDHint:
    """A hint extracted from compiler/validator feedback."""
    source: str          # "compiler" | "lint" | "schema"
    message: str         # The raw error message
    enhanced_prompt: str  # Original prompt + hint appended
    is_valid: bool       # Whether this is a usable hint (>10 chars)


def extract_compiler_hints(validation_result, original_prompt: str) -> list[OPDHint]:
    """
    Extract OPD hints from validation results.

    Unlike standard OPD which uses an LLM judge to extract hints,
    we use compiler/linter error messages directly.

    Args:
        validation_result: ValidationResult from HorizonEnvironment
        original_prompt: The original generation prompt

    Returns:
        List of OPDHint objects
    """
    hints = []

    # JSON parsing errors → direct hint
    if not validation_result.json_valid and validation_result.json_error:
        msg = validation_result.json_error
        if len(msg) > 10:
            hints.append(OPDHint(
                source="json",
                message=msg,
                enhanced_prompt=f"{original_prompt}\n\n[Fix Required]\n{msg}",
                is_valid=True,
            ))

    # Section reference errors → direct hint
    if not validation_result.sections_valid and validation_result.sections_error:
        msg = validation_result.sections_error
        if len(msg) > 10:
            hints.append(OPDHint(
                source="sections",
                message=msg,
                enhanced_prompt=f"{original_prompt}\n\n[Fix Required]\n{msg}",
                is_valid=True,
            ))

    # Theme check errors → each error becomes a hint
    if not validation_result.theme_check_valid:
        for err in validation_result.theme_check_errors:
            if len(err) > 10:
                hints.append(OPDHint(
                    source="theme_check",
                    message=err,
                    enhanced_prompt=f"{original_prompt}\n\n[Fix Required]\n{err}",
                    is_valid=True,
                ))

    return hints


def compute_opd_advantage(
    model,
    tokenizer,
    original_prompt: str,
    generated_response: str,
    hint: OPDHint,
    device: str = "cuda",
) -> list[float]:
    """
    Compute token-level OPD advantage.

    A[k] = log π_teacher(token_k | enhanced) - log π_student(token_k | original)

    Where:
      - π_teacher sees the compiler error hint (open-book exam)
      - π_student only sees the original prompt (closed-book)
      - Positive A[k] = teacher agrees with this token → reinforce
      - Negative A[k] = teacher disagrees → suppress

    Args:
        model: The language model
        tokenizer: The tokenizer
        original_prompt: Original prompt without hints
        generated_response: The model's response to evaluate
        hint: OPDHint containing the enhanced prompt
        device: Device to run on

    Returns:
        List of per-token advantages (same length as response tokens)
    """
    import torch

    # Tokenize response
    response_ids = tokenizer.encode(generated_response, add_special_tokens=False)

    # Build student input (original prompt + response)
    student_input = tokenizer(
        original_prompt + generated_response,
        return_tensors="pt",
    ).to(device)

    # Build teacher input (enhanced prompt + response)
    teacher_input = tokenizer(
        hint.enhanced_prompt + generated_response,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # Get student log probs
        student_logits = model(**student_input).logits
        student_log_probs = torch.log_softmax(student_logits, dim=-1)

        # Get teacher log probs
        teacher_logits = model(**teacher_input).logits
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    # Extract log probs for the actual response tokens
    # Align: find where response tokens start in each input
    student_prompt_len = len(tokenizer.encode(original_prompt, add_special_tokens=False))
    teacher_prompt_len = len(tokenizer.encode(hint.enhanced_prompt, add_special_tokens=False))

    advantages = []
    for i, token_id in enumerate(response_ids):
        s_idx = student_prompt_len + i
        t_idx = teacher_prompt_len + i

        if s_idx < student_log_probs.shape[1] and t_idx < teacher_log_probs.shape[1]:
            s_logp = student_log_probs[0, s_idx, token_id].item()
            t_logp = teacher_log_probs[0, t_idx, token_id].item()
            advantages.append(t_logp - s_logp)
        else:
            advantages.append(0.0)

    return advantages


def compiler_opd_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that combines binary outcome with OPD signal direction.

    For use with GRPOTrainer: returns scalar reward per sample.
    The token-level OPD advantages are stored in kwargs for the trainer
    to use during gradient computation.

    For now, returns binary reward + small bonus based on hint quality.
    Full token-level OPD integration requires custom trainer modification.
    """
    env = kwargs["env"]
    template_types = kwargs["template_types"]
    responses = [c[0]["content"] for c in completions]
    rewards = []

    opd_hints_list = kwargs.get("_opd_hints_collector", None)

    for response, tpl_type, prompt in zip(responses, template_types, prompts):
        result = env.validate(tpl_type, response)

        # Base binary reward
        base_reward = 1.0 if result.all_passed else 0.0

        # Extract compiler hints for OPD (stored for trainer)
        if not result.all_passed:
            prompt_text = prompt[0]["content"] if isinstance(prompt, list) else prompt
            hints = extract_compiler_hints(result, prompt_text)
            if opd_hints_list is not None:
                opd_hints_list.append(hints)
        else:
            if opd_hints_list is not None:
                opd_hints_list.append([])

        rewards.append(base_reward)

    return rewards
