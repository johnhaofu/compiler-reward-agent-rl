"""
Agent Tools for Horizon Theme Coding Agent.

8 tools: list_files, read_file, grep, write_file, list_components, get_section_schema, validate, done
"""

import json
import os
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, field

from environments.sitemuse_validator import SitemuseValidator


# ── Tool Definitions (OpenAI function calling format) ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory within the Horizon theme. Use to discover available sections, blocks, templates, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the theme. Examples: 'sections/', 'blocks/', 'templates/', 'snippets/'"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the Horizon theme. Use to study existing templates, section code, and code patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file. Examples: 'templates/page.json', 'sections/hero.liquid'"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern across files in the theme. Supports regex. Use to find how specific settings, block types, or patterns are used. Case-insensitive. Can search a directory or a single file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Examples: 'sections/', 'templates/', 'blocks/'",
                        "default": "."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace. Use to create or update template JSON files. Only writes to the workspace, not the base theme.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path. Examples: 'templates/page.about.json'"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full file content to write"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_components",
            "description": "List all available section types and block types in the Horizon theme with brief descriptions. Use before generating templates to know what components are available.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_section_schema",
            "description": "Get the full schema definition of a specific section, including all settings and accepted block types. Use to understand what settings and blocks a section supports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "description": "Section name. Examples: 'hero', 'main-page', 'main-collection', 'product-information'"
                    }
                },
                "required": ["section_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate",
            "description": "Validate theme files by upserting to Shopify API. Returns detailed error messages if validation fails. Always validate after writing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to validate. Examples: ['templates/page.about.json']"
                    }
                },
                "required": ["files"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the task is complete. Call after validation passes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                        "description": "Whether the task was completed successfully"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was done"
                    }
                },
                "required": ["status"]
            }
        }
    },
]


# ── System Prompt ──

SYSTEM_PROMPT = """You are an expert Shopify theme developer working with the Horizon theme.

## Your Task
Generate valid Shopify theme template files based on the user's requirements.
The templates must be compatible with the Horizon theme and pass Shopify's validation.

## Available Tools
- **list_files**: Browse theme directories to see available sections, blocks, templates
- **read_file**: Read existing files to understand patterns and schemas
- **grep**: Search across files for specific patterns or settings usage
- **write_file**: Create or update files in the workspace
- **list_components**: Get all available section and block types with descriptions
- **get_section_schema**: Get detailed schema for a specific section (settings + block types)
- **validate**: Check files against Shopify's theme validation API
- **done**: Signal task completion

## Workflow
1. **Research**: Use list_components to see available sections. Use get_section_schema to understand section settings. Use read_file to study existing templates as reference.
2. **Generate**: Use write_file to create the template JSON.
3. **Validate**: Use validate to check. If errors, read them carefully and fix.
4. **Complete**: Call done when validation passes.

## Rules
- Template JSON must have "sections" and "order" keys
- Only use section types that exist in the theme (check with list_components)
- Only use block types that sections accept (check with get_section_schema)
- Output valid JSON in write_file — no comments, no markdown
- Always validate before calling done
- If validation fails, fix based on the error message and validate again
"""


# ── Tool Runtime ──

@dataclass
class ToolCall:
    """Record of a tool call and its result."""
    turn: int
    tool_name: str
    arguments: dict
    result: str
    is_error: bool = False


@dataclass
class AgentWorkspace:
    """
    Manages the agent's tool execution environment.

    Base theme (Horizon) is read-only.
    Workspace files (agent-generated) are writable.
    """
    horizon_path: str
    workspace_dir: str = ""
    components_path: str = "data/horizon_components.json"
    schemas_dir: str = "data/schemas"
    validator: SitemuseValidator = None
    tool_history: list[ToolCall] = field(default_factory=list)
    workspace_files: dict = field(default_factory=dict)
    _components_cache: dict = None
    _done: bool = False
    _done_status: str = ""

    def __post_init__(self):
        if not self.validator:
            self.validator = SitemuseValidator()
        if not self.workspace_dir:
            import tempfile
            self.workspace_dir = tempfile.mkdtemp(prefix="agent_workspace_")

    @property
    def is_done(self) -> bool:
        return self._done

    def execute_tool(self, turn: int, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as a string."""
        handler = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "grep": self._grep,
            "write_file": self._write_file,
            "list_components": self._list_components,
            "get_section_schema": self._get_section_schema,
            "validate": self._validate,
            "done": self._done_tool,
        }.get(tool_name)

        if not handler:
            result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            is_error = True
        else:
            try:
                result = handler(**arguments)
                is_error = False
            except Exception as e:
                result = json.dumps({"error": str(e)})
                is_error = True

        call = ToolCall(turn=turn, tool_name=tool_name, arguments=arguments,
                        result=result, is_error=is_error)
        self.tool_history.append(call)
        return result

    def _list_files(self, path: str) -> str:
        """List files in a theme directory."""
        # Check workspace first, then base theme
        full_path = Path(self.horizon_path) / path
        if not full_path.exists():
            return json.dumps({"error": f"Directory not found: {path}"})

        files = sorted([
            f.name for f in full_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])

        # Also include workspace files in this directory
        for wp, _ in self.workspace_files.items():
            if wp.startswith(path):
                wf = wp[len(path):].split("/")[0]
                if wf and wf not in files:
                    files.append(f"{wf} (workspace)")

        return json.dumps(files)

    def _read_file(self, path: str) -> str:
        """Read a file from workspace or base theme."""
        # Check workspace first
        if path in self.workspace_files:
            return self.workspace_files[path]

        # Then base theme
        full_path = Path(self.horizon_path) / path
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {path}"})

        content = full_path.read_text()
        # Truncate very long files
        if len(content) > 8000:
            content = content[:8000] + "\n... (truncated, file too long)"
        return content

    def _grep(self, pattern: str, path: str = ".") -> str:
        """Search for pattern in files."""
        results = []
        search_path = Path(self.horizon_path) / path

        if not search_path.exists():
            # Also check workspace
            ws_path = Path(self.workspace_dir) / path if self.workspace_dir else None
            if ws_path and ws_path.exists():
                search_path = ws_path
            else:
                return json.dumps({"error": f"Path not found: {path}"})

        # Handle both single file and directory
        if search_path.is_file():
            files_to_search = [search_path]
        else:
            files_to_search = list(search_path.rglob("*"))

        # Compile regex (fall back to literal match if invalid regex)
        try:
            import re
            regex = re.compile(pattern, re.IGNORECASE)
            use_regex = True
        except re.error:
            use_regex = False

        for f in files_to_search:
            if f.is_file() and f.suffix in (".liquid", ".json"):
                try:
                    content = f.read_text()
                    for i, line in enumerate(content.split("\n"), 1):
                        matched = regex.search(line) if use_regex else (pattern.lower() in line.lower())
                        if matched:
                            try:
                                rel = str(f.relative_to(Path(self.horizon_path)))
                            except ValueError:
                                rel = str(f)
                            results.append(f"{rel}:{i}: {line.strip()[:120]}")
                except Exception:
                    continue

        if not results:
            return json.dumps({"matches": 0, "message": f"No matches for '{pattern}'"})

        # Limit results
        total = len(results)
        results = results[:20]
        return json.dumps({"matches": total, "results": results})

    def _write_file(self, path: str, content: str) -> str:
        """Write file to workspace."""
        # Security: only allow templates/ and sections/ paths
        allowed_prefixes = ("templates/", "sections/", "blocks/", "snippets/")
        if not any(path.startswith(p) for p in allowed_prefixes):
            return json.dumps({"error": f"Can only write to {allowed_prefixes}"})

        self.workspace_files[path] = content

        # Also write to disk for validate to read
        full_path = Path(self.workspace_dir) / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        return json.dumps({"success": True, "path": path, "size": len(content)})

    def _list_components(self) -> str:
        """List all available sections and blocks."""
        if self._components_cache is None:
            comp_path = Path(self.components_path)
            if comp_path.exists():
                self._components_cache = json.loads(comp_path.read_text())
            else:
                return json.dumps({"error": "Components data not found"})

        # Build concise summary
        sections = []
        for name, info in self._components_cache.get("sections", {}).items():
            block_types = [b["type"] for b in info.get("block_types", [])[:5]]
            sections.append({
                "name": name,
                "display_name": info.get("name", name),
                "settings_count": info.get("settings_count", 0),
                "block_types": block_types,
            })

        blocks = self._components_cache.get("blocks", [])

        return json.dumps({
            "sections": sections,
            "blocks": blocks,
            "total_sections": len(sections),
            "total_blocks": len(blocks),
        }, ensure_ascii=False)

    def _get_section_schema(self, section_name: str) -> str:
        """Get detailed schema for a section."""
        schema_path = Path(self.schemas_dir) / f"{section_name}.json"
        if schema_path.exists():
            schema = json.loads(schema_path.read_text())
            return json.dumps(schema, ensure_ascii=False)

        # Try with underscore prefix
        schema_path = Path(self.schemas_dir) / f"_{section_name}.json"
        if schema_path.exists():
            schema = json.loads(schema_path.read_text())
            return json.dumps(schema, ensure_ascii=False)

        return json.dumps({
            "error": f"Schema not found for section '{section_name}'. "
                     f"Use list_components to see available sections."
        })

    def _validate(self, files: list[str]) -> str:
        """Validate files via Sitemuse API."""
        all_errors = []
        all_passed = []

        for file_path in files:
            # Read from workspace
            content = self.workspace_files.get(file_path)
            if not content:
                all_errors.append({
                    "file": file_path,
                    "message": f"File not found in workspace. Use write_file first."
                })
                continue

            # Pass file_path directly — validator handles .json vs .liquid
            result = self.validator.validate(file_path, content)

            if result.all_passed:
                all_passed.append(file_path)
            else:
                for err in result.get_all_errors():
                    all_errors.append({"file": file_path, "message": err})

        passed = len(all_errors) == 0
        response = {"passed": passed}
        if all_passed:
            response["files_validated"] = all_passed
        if all_errors:
            response["errors"] = all_errors

        return json.dumps(response)

    def _done_tool(self, status: str, summary: str = "") -> str:
        """Signal task completion."""
        self._done = True
        self._done_status = status
        return json.dumps({"status": status, "summary": summary})

    def get_metrics(self) -> dict:
        """
        Extract episode metrics aligned with SWE-bench / FeatureBench standards.

        ═══ Core (aligned with SWE-bench) ═══
          resolved         — Task fully resolved (done=success + last validate passed)
          first_try_valid  — First validate passed (code quality without fix)
          fix_rate         — First validate failed → eventually resolved (compiler feedback value)
                             ★ This is our paper's key metric

        ═══ Efficiency (aligned with FeatureBench) ═══
          total_turns      — Total tool calls
          validate_attempts — Number of validate calls
          turns_to_resolve — Tool calls until first passing validate (None if never passed)
          fix_turns        — Tool calls spent on fixing (between first fail and first pass)
          research_turns   — Research tool calls before first write
          token_input      — Total input tokens (set externally by eval script)
          token_output     — Total output tokens (set externally)

        ═══ Analysis ═══
          tool_sequence    — Ordered tool call names
          error_trace      — Per-validate errors (for error type analysis)
          final_files      — Generated file contents (for inspection)
        """
        validate_calls = [t for t in self.tool_history if t.tool_name == "validate"]
        write_calls = [t for t in self.tool_history if t.tool_name == "write_file"]
        research_tools = {"list_files", "read_file", "grep", "list_components", "get_section_schema"}

        # ── Phase analysis ──
        first_write_turn = write_calls[0].turn if write_calls else None

        research_turns = sum(
            1 for t in self.tool_history
            if t.tool_name in research_tools
            and (first_write_turn is None or t.turn < first_write_turn)
        )

        # ── Validate analysis ──
        validate_results = []
        for vc in validate_calls:
            parsed = json.loads(vc.result)
            passed = parsed.get("passed", False)
            errors = [e.get("message", "") for e in parsed.get("errors", [])]
            validate_results.append({"turn": vc.turn, "passed": passed, "errors": errors})

        first_try_valid = validate_results[0]["passed"] if validate_results else False
        last_validate_passed = validate_results[-1]["passed"] if validate_results else False

        # Resolved = done(success) AND last validate passed
        resolved = (self._done_status == "success" and last_validate_passed)

        # Fix rate = first fail → eventually resolved
        fix_rate = (not first_try_valid and resolved) if validate_results else False

        # Turns to resolve
        turns_to_resolve = None
        for vr in validate_results:
            if vr["passed"]:
                turns_to_resolve = vr["turn"] + 1
                break

        # Fix turns: between first failed validate and first passing validate
        fix_turns = 0
        if fix_rate and turns_to_resolve is not None:
            first_fail_turn = validate_results[0]["turn"]
            fix_turns = sum(
                1 for t in self.tool_history
                if first_fail_turn < t.turn < turns_to_resolve
            )

        # ── Error trace ──
        error_trace = []
        all_error_messages = []
        error_types = {}
        for i, vr in enumerate(validate_results):
            if vr["errors"]:
                error_trace.append({
                    "attempt": i + 1,
                    "turn": vr["turn"],
                    "passed": vr["passed"],
                    "errors": vr["errors"],
                })
            for err in vr["errors"]:
                all_error_messages.append(err)
                # Classify error type
                if "Invalid JSON" in err:
                    etype = "json_syntax"
                elif "missing required key" in err:
                    etype = "missing_key"
                elif "unknown key" in err:
                    etype = "unknown_key"
                elif "does not refer to an existing section" in err:
                    etype = "invalid_section_type"
                elif "block type must be defined" in err or "Invalid value for type in block" in err:
                    etype = "invalid_block_type"
                elif "must exist in sections" in err:
                    etype = "order_mismatch"
                elif "Liquid syntax error" in err:
                    etype = "liquid_syntax"
                elif "Invalid JSON in tag" in err:
                    etype = "schema_json"
                else:
                    etype = "other"
                error_types[etype] = error_types.get(etype, 0) + 1

        # ── Tool usage breakdown ──
        tool_counts = {}
        for t in self.tool_history:
            tool_counts[t.tool_name] = tool_counts.get(t.tool_name, 0) + 1

        # ── Final files ──
        final_files = dict(self.workspace_files)

        return {
            # Core (SWE-bench aligned)
            "resolved": resolved,
            "first_try_valid": first_try_valid,
            "fix_rate": fix_rate,

            # Efficiency (FeatureBench aligned)
            "total_turns": len(self.tool_history),
            "validate_attempts": len(validate_calls),
            "write_calls": len(write_calls),
            "turns_to_resolve": turns_to_resolve,
            "fix_turns": fix_turns,
            "research_turns": research_turns,
            "done_status": self._done_status,

            # Analysis
            "tool_sequence": [t.tool_name for t in self.tool_history],
            "tool_counts": tool_counts,
            "error_trace": error_trace,
            "error_types": error_types,
            "errors_encountered": all_error_messages,
            "unique_errors": list(set(all_error_messages)),

            # Output
            "final_files": final_files,
        }

    def cleanup(self):
        """Clean up workspace directory."""
        import shutil
        if self.workspace_dir and Path(self.workspace_dir).exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
