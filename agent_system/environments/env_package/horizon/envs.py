"""
Horizon Theme Environment for verl-agent.

Gym-style multi-turn environment for Shopify Horizon template generation.
Each step: model outputs a tool call → env executes → returns observation.
"""

import json
import concurrent.futures
import asyncio
from typing import Any, Dict, List
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
from omegaconf import DictConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from environments.tools import AgentWorkspace, TOOLS


# Tool descriptions for the model prompt
TOOL_DESCRIPTIONS = "\n".join(
    f"- {t['function']['name']}: {t['function']['description']}"
    for t in TOOLS
)


class HorizonSingleEnv:
    """Single Horizon theme environment instance."""

    def __init__(self, horizon_path: str, schemas_dir: str = "data/schemas",
                 components_path: str = "data/horizon_components.json"):
        self.horizon_path = horizon_path
        self.schemas_dir = schemas_dir
        self.components_path = components_path
        self.workspace = None
        self.turn = 0
        self.max_turns = 50
        self.done = False

    def reset(self, extras: dict):
        """Reset environment with a new task."""
        self.workspace = AgentWorkspace(
            horizon_path=self.horizon_path,
            components_path=self.components_path,
            schemas_dir=self.schemas_dir,
        )
        self.turn = 0
        self.max_turns = extras.get("max_turns", 50)
        self.done = False

    def step(self, action: str) -> dict:
        """Execute a tool call action.

        Args:
            action: Model-generated text containing a tool call.
                    Expected format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
                    Or JSON: {"name": "tool_name", "arguments": {...}}

        Returns:
            dict with keys: observations, reward, done, metadata
        """
        if self.done or self.turn >= self.max_turns:
            return {
                "observations": "Episode already ended.",
                "reward": 0.0,
                "done": True,
                "metadata": {"won": False},
            }

        # Parse tool call from model output
        tool_name, arguments = self._parse_action(action)

        if tool_name is None:
            self.turn += 1
            return {
                "observations": "Invalid action format. Use a tool call.",
                "reward": 0.0,
                "done": self.turn >= self.max_turns,
                "metadata": {"won": False, "is_action_valid": False},
            }

        # Execute tool
        result = self.workspace.execute_tool(
            turn=self.turn,
            tool_name=tool_name,
            arguments=arguments,
        )
        self.turn += 1

        # Check if done
        is_done = self.workspace.is_done or self.turn >= self.max_turns

        # Compute reward (only at end of episode)
        reward = 0.0
        won = False
        if is_done:
            metrics = self.workspace.get_metrics()
            if metrics["resolved"]:
                reward = 1.0
                won = True
            elif metrics["first_try_valid"]:
                reward = 0.5
            self.done = True
            self.workspace.cleanup()

        return {
            "observations": result,
            "reward": reward,
            "done": is_done,
            "metadata": {
                "won": won,
                "tool_name": tool_name,
                "turn": self.turn,
                "is_action_valid": True,
            },
        }

    def _parse_action(self, text: str):
        """Parse tool call from model output.

        Supports two formats:
        1. Qwen3.5 format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        2. JSON format: {"name": "tool_name", "arguments": {...}}
        """
        text = text.strip()

        # Try Qwen3.5 <tool_call> format
        if "<tool_call>" in text:
            return self._parse_qwen_tool_call(text)

        # Try JSON format
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "name" in data:
                return data["name"], data.get("arguments", {})
        except (json.JSONDecodeError, KeyError):
            pass

        # Try to find function call pattern in text
        for tool in TOOLS:
            name = tool["function"]["name"]
            if name + "(" in text or f'"{name}"' in text:
                # Try to extract arguments
                try:
                    start = text.index("{")
                    end = text.rindex("}") + 1
                    args = json.loads(text[start:end])
                    return name, args
                except (ValueError, json.JSONDecodeError):
                    return name, {}

        return None, {}

    def _parse_qwen_tool_call(self, text: str):
        """Parse Qwen3.5 tool call format."""
        import re

        # Extract function name
        func_match = re.search(r"<function=([^>]+)>", text)
        if not func_match:
            return None, {}
        func_name = func_match.group(1)

        # Extract parameters
        args = {}
        param_pattern = re.findall(
            r"<parameter=([^>]+)>(.*?)</parameter>", text, re.DOTALL
        )
        for param_name, param_value in param_pattern:
            param_value = param_value.strip()
            # Try parsing as JSON value
            try:
                args[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                args[param_name] = param_value

        return func_name, args

    def close(self):
        if self.workspace:
            self.workspace.cleanup()


class HorizonMultiProcessEnv(gym.Env):
    """Vectorized Horizon environment for verl-agent."""

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: DictConfig = None,
    ):
        super().__init__()

        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.max_steps = env_config.max_steps

        horizon_path = env_config.get("horizon_path", "/root/autodl-tmp/horizon")
        schemas_dir = env_config.get("schemas_dir", "data/schemas")
        components_path = env_config.get("components_path", "data/horizon_components.json")

        self.envs = [
            HorizonSingleEnv(
                horizon_path=horizon_path,
                schemas_dir=schemas_dir,
                components_path=components_path,
            )
            for _ in range(self.batch_size)
        ]

        max_workers = min(self.batch_size, 32)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()

    def _sync_reset(self, env, kwargs):
        extras = {
            "max_turns": self.max_steps,
            "ground_truth": kwargs.get("ground_truth", ""),
        }
        env.reset(extras)
        obs = kwargs.get("question", kwargs.get("prompt", ""))
        info = {"data_source": kwargs.get("data_source", "horizon")}
        return obs, info

    def _sync_step(self, env, action: str):
        out = env.step(action)
        obs = out["observations"]
        reward = out["reward"]
        done = out["done"]
        info = dict(out.get("metadata", {}))
        info["won"] = bool(done and reward >= 1.0)
        return obs, reward, done, info

    def reset(self, kwargs: List[Dict]):
        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {"question": "", "ground_truth": "", "data_source": "dummy"}
        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))
        obs_list = [o for o, k in zip(obs_list, valid_mask) if k]
        info_list = [i for i, k in zip(info_list, valid_mask) if k]
        return obs_list, info_list

    def step(self, actions: List[str]):
        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))
        obs_list = [o for o, k in zip(obs_list, valid_mask) if k]
        reward_list = [r for r, k in zip(reward_list, valid_mask) if k]
        done_list = [d for d, k in zip(done_list, valid_mask) if k]
        info_list = [i for i, k in zip(info_list, valid_mask) if k]

        return obs_list, reward_list, done_list, info_list

    def close(self):
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()


def build_horizon_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None,
):
    return HorizonMultiProcessEnv(
        seed=seed, env_num=env_num, group_n=group_n,
        is_train=is_train, env_config=env_config,
    )
