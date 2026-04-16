"""
Projection function for Horizon environment.

Maps model text output → (action_string, valid_flag).
The action_string is passed directly to HorizonSingleEnv.step() which parses the tool call.
"""

from typing import List, Tuple
import numpy as np


def horizon_projection(text_actions: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Identity projection — pass model text directly to environment.
    The environment handles parsing tool calls from text.

    Returns:
        actions: Same as input text_actions
        valids: 1.0 for non-empty actions, 0.0 for empty
    """
    actions = []
    valids = []
    for text in text_actions:
        text = text.strip()
        if text:
            actions.append(text)
            valids.append(1.0)
        else:
            actions.append("")
            valids.append(0.0)
    return actions, np.array(valids)
