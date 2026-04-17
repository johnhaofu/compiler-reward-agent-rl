"""
Horizon Theme Tools for verl-agent.

Wraps AgentWorkspace as verl BaseTool instances. Each tool maps to one of the
9 agent tools (list_files, read_file, grep, write_file, edit_json,
list_components, get_section_schema, validate, done).

Reward design:
- Step rewards: small (validate pass=+0.5, errors=-0.1, done success=+0.5)
- Final episode reward (calc_reward): resolved=1.0, first_try_valid=0.5, else=0.0
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

# Make project root importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from environments.tools import AgentWorkspace


# Per-trajectory workspace storage. Keyed by instance_id which SGLang
# assigns once per rollout episode and reuses across all 9 tool calls
# within that episode.
_workspaces: dict[str, AgentWorkspace] = {}
_turn_counters: dict[str, int] = {}


def _get_workspace(instance_id: str, config: dict) -> AgentWorkspace:
    """Get or create workspace for this trajectory."""
    if instance_id not in _workspaces:
        _workspaces[instance_id] = AgentWorkspace(
            horizon_path=config.get("horizon_path", os.environ.get(
                "HORIZON_PATH", "/root/autodl-tmp/horizon")),
            components_path=config.get("components_path",
                                       "data/horizon_components.json"),
            schemas_dir=config.get("schemas_dir", "data/schemas"),
        )
        _turn_counters[instance_id] = 0
    return _workspaces[instance_id]


class HorizonTool(BaseTool):
    """One BaseTool subclass per Horizon tool name (config controls dispatch)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.tool_name = tool_schema.function.name
        self._tool_config = config or {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        instance_id = await super().create(instance_id, **kwargs)
        # Pre-create workspace so all tools share state for this trajectory
        _get_workspace(instance_id, self._tool_config)
        return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        workspace = _get_workspace(instance_id, self._tool_config)
        turn = _turn_counters.get(instance_id, 0)
        _turn_counters[instance_id] = turn + 1

        try:
            result = workspace.execute_tool(
                turn=turn,
                tool_name=self.tool_name,
                arguments=parameters,
            )
        except Exception as e:
            return f"Tool error: {e}", -0.1, {"tool_error": str(e)}

        # Step-level rewards (small) — main reward comes from calc_reward
        step_reward = 0.0
        metrics = {"tool": self.tool_name, "turn": turn}

        if self.tool_name == "validate":
            try:
                res = json.loads(result)
                if res.get("passed"):
                    step_reward = 0.3  # encourage successful validation
                metrics["validation_passed"] = res.get("passed", False)
            except (json.JSONDecodeError, TypeError):
                pass
        elif self.tool_name == "done":
            try:
                res = json.loads(result)
                if res.get("status") == "success":
                    step_reward = 0.3
                metrics["done_success"] = res.get("status") == "success"
            except (json.JSONDecodeError, TypeError):
                pass

        # Penalize tool errors
        if isinstance(result, str) and result.startswith('{"error"'):
            step_reward = min(step_reward, -0.05)

        return result, step_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Final episode reward from workspace metrics.

        Returns:
            1.0  - resolved (all validations pass + done called)
            0.5  - first_try_valid (validation passed without retry)
            0.0  - failed
        """
        workspace = _workspaces.get(instance_id)
        if workspace is None:
            return 0.0

        try:
            metrics = workspace.get_metrics()
            if metrics.get("resolved"):
                return 1.0
            if metrics.get("first_try_valid"):
                return 0.5
        except Exception:
            pass
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Cleanup workspace at end of episode."""
        workspace = _workspaces.pop(instance_id, None)
        _turn_counters.pop(instance_id, None)
        if workspace:
            try:
                workspace.cleanup()
            except Exception:
                pass
