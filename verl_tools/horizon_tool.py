"""
Horizon Theme Tools for verl-agent.

Wraps AgentWorkspace as verl BaseTool instances.
Each tool maps to one of the 9 agent tools (list_files, read_file, etc.).
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from environments.tools import AgentWorkspace

# Shared workspace instances per trajectory
_workspaces: dict[str, AgentWorkspace] = {}

HORIZON_PATH = "/root/autodl-tmp/horizon"
SCHEMAS_DIR = "data/schemas"
COMPONENTS_PATH = "data/horizon_components.json"


def _get_workspace(instance_id: str) -> AgentWorkspace:
    if instance_id not in _workspaces:
        _workspaces[instance_id] = AgentWorkspace(
            horizon_path=HORIZON_PATH,
            components_path=COMPONENTS_PATH,
            schemas_dir=SCHEMAS_DIR,
        )
    return _workspaces[instance_id]


class HorizonTool(BaseTool):
    """Single tool that dispatches to AgentWorkspace based on tool name."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.tool_name = tool_schema.function.name
        self._turn_counters: dict[str, int] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        instance_id = await super().create(instance_id, **kwargs)
        _get_workspace(instance_id)  # Pre-create workspace
        self._turn_counters[instance_id] = 0
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        workspace = _get_workspace(instance_id)
        turn = self._turn_counters.get(instance_id, 0)
        self._turn_counters[instance_id] = turn + 1

        result = workspace.execute_tool(
            turn=turn,
            tool_name=self.tool_name,
            arguments=parameters,
        )

        # Step reward: small bonus for valid tool use, penalty for errors
        step_reward = 0.0
        metrics = {}

        if self.tool_name == "validate":
            try:
                res = json.loads(result)
                if res.get("passed"):
                    step_reward = 0.5  # Validation passed
                metrics["validation_passed"] = res.get("passed", False)
            except (json.JSONDecodeError, TypeError):
                pass
        elif self.tool_name == "done":
            try:
                res = json.loads(result)
                if res.get("status") == "success":
                    step_reward = 0.5
                metrics["done_success"] = res.get("status") == "success"
            except (json.JSONDecodeError, TypeError):
                pass

        if "error" in result.lower()[:50]:
            step_reward = -0.1

        return result, step_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Final episode reward based on workspace metrics."""
        workspace = _workspaces.get(instance_id)
        if workspace is None:
            return 0.0

        metrics = workspace.get_metrics()
        if metrics["resolved"]:
            return 1.0
        elif metrics["first_try_valid"]:
            return 0.5
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        workspace = _workspaces.pop(instance_id, None)
        if workspace:
            workspace.cleanup()
        self._turn_counters.pop(instance_id, None)
