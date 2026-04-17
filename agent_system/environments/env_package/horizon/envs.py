"""
Horizon Theme Environment for verl-agent.

Stub env_manager — actual tool execution is handled by HorizonTool (BaseTool).
This env only:
1. Provides initial task prompt at reset()
2. Returns final reward when SGLang rollout finishes

Multi-turn rollout flow (verl-agent + SGLang + BaseTool):
  reset()   → task prompt
  SGLang internally handles tool calls via HorizonTool.execute()
  When agent calls "done": episode ends, reward = HorizonTool.calc_reward()
"""

import asyncio
import concurrent.futures
from typing import Dict, List, Optional

import gym
from omegaconf import DictConfig


class HorizonSingleEnv:
    """Single Horizon task — provides prompt, computes final reward."""

    def __init__(self):
        self.task = None
        self.done = False

    def reset(self, extras: dict):
        """Store task description."""
        self.task = extras.get("question", "")
        self.done = False

    def step(self, action: str) -> dict:
        """No-op step (real work done by SGLang+BaseTool).

        Called only if multi_turn rollout is disabled.
        With multi_turn enabled, SGLang handles the rollout end-to-end.
        """
        self.done = True
        return {
            "observations": "",
            "reward": 0.0,
            "done": True,
            "metadata": {"won": False},
        }

    def close(self):
        pass


class HorizonMultiProcessEnv(gym.Env):
    """Vectorized Horizon environment for verl-agent."""

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.max_steps = env_config.max_steps if env_config else 50

        self.envs = [HorizonSingleEnv() for _ in range(self.batch_size)]

        max_workers = min(self.batch_size, 32)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()

    def _sync_reset(self, env, kwargs):
        extras = {
            "max_turns": self.max_steps,
            "question": kwargs.get("question", kwargs.get("prompt", "")),
            "ground_truth": kwargs.get("ground_truth", ""),
        }
        env.reset(extras)
        info = {"data_source": kwargs.get("data_source", "horizon")}
        return extras["question"], info

    def _sync_step(self, env, action: str):
        out = env.step(action)
        return out["observations"], out["reward"], out["done"], dict(out.get("metadata", {}))

    def reset(self, kwargs: List[Dict]):
        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {"question": "", "ground_truth": "", "data_source": "dummy"}
        padded = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))
        obs_list = [o for o, k in zip(obs_list, valid_mask) if k]
        info_list = [i for i, k in zip(info_list, valid_mask) if k]
        return obs_list, info_list

    def step(self, actions: List[str]):
        pad_n = self.batch_size - len(actions)
        padded = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded)
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


def build_horizon_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=None):
    return HorizonMultiProcessEnv(
        seed=seed, env_num=env_num, group_n=group_n,
        is_train=is_train, env_config=env_config,
    )
