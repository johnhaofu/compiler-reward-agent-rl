"""
Verify Engine: Check if generated output meets task requirements.

Supports two verify types:
  1. json_check  — Check JSON template against path-based rules
  2. multi_file  — Check multiple files (liquid + json) against content/structure rules

This is Layer 2 verification (requirements check), complementing:
  Layer 1: Shopify API validation (structural correctness)
  Layer 3: Playwright screenshot + VLM (visual quality)
"""

import json
import re
from typing import Any


def run_verify(verify_spec: dict, final_files: dict) -> dict:
    """
    Run verification checks on generated files.

    Args:
        verify_spec: The "verify" field from eval task definition
        final_files: Dict of {path: content} from AgentWorkspace

    Returns:
        {
            "passed": bool,
            "total_checks": int,
            "passed_checks": int,
            "failed_checks": list of {"check": ..., "reason": ...},
        }
    """
    verify_type = verify_spec.get("type", "json_check")

    if verify_type == "json_check":
        return _verify_json_checks(verify_spec.get("checks", []), final_files)
    elif verify_type == "multi_file":
        return _verify_multi_file(verify_spec.get("checks", {}), final_files)
    else:
        return {"passed": False, "total_checks": 0, "passed_checks": 0,
                "failed_checks": [{"check": "type", "reason": f"Unknown verify type: {verify_type}"}]}


def _verify_json_checks(checks: list, final_files: dict) -> dict:
    """Verify JSON template files against path-based rules."""
    # Find the template JSON file
    json_files = {k: v for k, v in final_files.items() if k.endswith(".json")}

    if not json_files:
        return {"passed": False, "total_checks": len(checks), "passed_checks": 0,
                "failed_checks": [{"check": "file_exists", "reason": "No JSON template file found in workspace"}]}

    # Parse all JSON files
    parsed_files = {}
    for path, content in json_files.items():
        try:
            parsed_files[path] = json.loads(content)
        except json.JSONDecodeError:
            parsed_files[path] = None

    results = []
    for check in checks:
        passed, reason = _run_single_check(check, parsed_files, final_files)
        results.append({"check": check, "passed": passed, "reason": reason})

    passed_count = sum(1 for r in results if r["passed"])
    failed = [{"check": r["check"], "reason": r["reason"]} for r in results if not r["passed"]]

    return {
        "passed": len(failed) == 0,
        "total_checks": len(checks),
        "passed_checks": passed_count,
        "failed_checks": failed,
    }


def _run_single_check(check: dict, parsed_files: dict, raw_files: dict) -> tuple[bool, str]:
    """Run a single verification check against all parsed JSON files."""
    path_pattern = check.get("path", "")

    # Collect all values matching the path pattern across all files
    values = []
    for fpath, data in parsed_files.items():
        if data is None:
            continue
        values.extend(_resolve_path(data, path_pattern))

    # ── Check types ──

    if "contains" in check:
        target = check["contains"]
        for v in values:
            if isinstance(v, str) and target in v:
                return True, ""
        # Also check raw file content
        for content in raw_files.values():
            if target in content:
                return True, ""
        return False, f"No value contains '{target}'"

    if "equals" in check:
        target = check["equals"]
        for v in values:
            if v == target:
                return True, ""
        return False, f"No value equals {target!r}, found: {values[:3]}"

    if "has_value" in check:
        target = check["has_value"]
        for v in values:
            if v == target:
                return True, ""
            if isinstance(v, list) and target in v:
                return True, ""
        return False, f"'{target}' not found in values: {values[:5]}"

    if "not_has_value" in check:
        target = check["not_has_value"]
        for v in values:
            if v == target:
                return False, f"Found unwanted value '{target}'"
            if isinstance(v, list) and target in v:
                return False, f"Found unwanted value '{target}' in list"
        return True, ""

    if "length_equals" in check:
        target = check["length_equals"]
        for v in values:
            if isinstance(v, (list, dict)) and len(v) == target:
                return True, ""
        return False, f"No collection has length {target}"

    if "length_gte" in check:
        target = check["length_gte"]
        for v in values:
            if isinstance(v, (list, dict)) and len(v) >= target:
                return True, ""
        return False, f"No collection has length >= {target}"

    if "min_count" in check:
        target = check["min_count"]
        for v in values:
            if isinstance(v, dict) and len(v) >= target:
                return True, ""
            if isinstance(v, list) and len(v) >= target:
                return True, ""
        return False, f"No collection has >= {target} items"

    if "value_before" in check:
        targets = check["value_before"]
        if len(targets) >= 2:
            for v in values:
                if isinstance(v, list):
                    # Check that first target appears before second
                    first_key = targets[0]
                    second_key = targets[1]
                    first_idx = _find_index_containing(v, first_key)
                    second_idx = _find_index_containing(v, second_key)
                    if first_idx is not None and second_idx is not None and first_idx < second_idx:
                        return True, ""
        return False, f"Order constraint {targets} not satisfied"

    if "last_contains" in check:
        target = check["last_contains"]
        for v in values:
            if isinstance(v, list) and v and target in str(v[-1]):
                return True, ""
        return False, f"Last element doesn't contain '{target}'"

    return False, f"Unknown check type: {check}"


def _verify_multi_file(checks: dict, final_files: dict) -> dict:
    """Verify multiple files (liquid + json) against content and structure rules."""
    all_results = []

    for file_path, file_checks in checks.items():
        content = final_files.get(file_path, "")
        if not content:
            all_results.append({
                "check": {"file": file_path}, "passed": False,
                "reason": f"File '{file_path}' not found in workspace"
            })
            continue

        for check in file_checks:
            passed, reason = _run_file_check(check, file_path, content, final_files)
            all_results.append({
                "check": {**check, "file": file_path},
                "passed": passed,
                "reason": reason,
            })

    passed_count = sum(1 for r in all_results if r["passed"])
    failed = [{"check": r["check"], "reason": r["reason"]} for r in all_results if not r["passed"]]

    return {
        "passed": len(failed) == 0,
        "total_checks": len(all_results),
        "passed_checks": passed_count,
        "failed_checks": failed,
    }


def _run_file_check(check: dict, file_path: str, content: str, all_files: dict) -> tuple[bool, str]:
    """Run a single check on a specific file."""

    # Content checks (for .liquid files)
    if "content_contains" in check:
        target = check["content_contains"]
        if target in content:
            return True, ""
        return False, f"File doesn't contain '{target}'"

    # JSON path checks (for .json files)
    if "json_path" in check:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return False, f"Cannot parse as JSON"

        values = _resolve_path(data, check["json_path"])

        if "has_value" in check:
            target = check["has_value"]
            for v in values:
                if v == target:
                    return True, ""
                if isinstance(v, list) and target in v:
                    return True, ""
            return False, f"'{target}' not found"

        if "contains" in check:
            target = check["contains"]
            for v in values:
                if isinstance(v, str) and target in v:
                    return True, ""
            for v in values:
                if target in str(v):
                    return True, ""
            return False, f"No value contains '{target}'"

        if "min_count" in check:
            target = check["min_count"]
            for v in values:
                if isinstance(v, (dict, list)) and len(v) >= target:
                    return True, ""
            return False, f"No collection has >= {target} items"

        if "length_gte" in check:
            target = check["length_gte"]
            for v in values:
                if isinstance(v, (list, dict)) and len(v) >= target:
                    return True, ""
            return False, f"No collection has length >= {target}"

        if "value_before" in check:
            targets = check["value_before"]
            for v in values:
                if isinstance(v, list) and len(targets) >= 2:
                    first_idx = _find_index_containing(v, targets[0])
                    second_idx = _find_index_containing(v, targets[1])
                    if first_idx is not None and second_idx is not None and first_idx < second_idx:
                        return True, ""
            return False, f"Order constraint not satisfied"

    return False, f"Unknown check: {check}"


# ── Path resolution helpers ──

def _resolve_path(data: Any, path: str) -> list:
    """
    Resolve a dot-separated path with wildcard support.

    Examples:
      "sections.main.settings.columns" → [3]
      "sections.*.type" → ["hero", "product-list"]
      "sections.*.blocks.*.settings.text" → ["<p>Hello</p>", ...]
      "order" → [["hero_abc", "product_list_def"]]
    """
    parts = path.split(".")
    return _resolve_recursive(data, parts)


def _resolve_recursive(data: Any, parts: list[str]) -> list:
    """Recursively resolve path parts."""
    if not parts:
        return [data]

    current = parts[0]
    rest = parts[1:]

    if current == "*":
        # Wildcard: iterate over all values
        results = []
        if isinstance(data, dict):
            for v in data.values():
                results.extend(_resolve_recursive(v, rest))
        elif isinstance(data, list):
            for item in data:
                results.extend(_resolve_recursive(item, rest))
        return results
    else:
        if isinstance(data, dict) and current in data:
            return _resolve_recursive(data[current], rest)
        return []


def _find_index_containing(lst: list, substring: str) -> int | None:
    """Find index of first element containing substring."""
    for i, item in enumerate(lst):
        if substring in str(item):
            return i
    return None


# ── Convenience ──

def verify_episode(eval_task: dict, episode_metrics: dict) -> dict:
    """
    Run verification for a complete episode.

    Args:
        eval_task: Task definition with "verify" field
        episode_metrics: Episode metrics with "final_files" field

    Returns:
        Verification result dict
    """
    verify_spec = eval_task.get("verify", {})
    final_files = episode_metrics.get("final_files", {})

    if not verify_spec:
        return {"passed": True, "total_checks": 0, "passed_checks": 0, "failed_checks": [],
                "note": "No verify spec defined"}

    return run_verify(verify_spec, final_files)


if __name__ == "__main__":
    # Quick test
    print("=== Test json_check ===")
    files = {
        "templates/index.json": json.dumps({
            "sections": {
                "hero_abc": {"type": "hero", "blocks": {"t1": {"type": "text", "settings": {"text": "<p>Summer Collection 2026</p>"}}, "b1": {"type": "button", "settings": {"label": "Shop Summer"}}}, "settings": {"color_scheme": "scheme-2"}},
                "pl_def": {"type": "product-list", "settings": {"max_products": 8}},
            },
            "order": ["hero_abc", "pl_def"]
        })
    }

    # Test L1-01 verify
    spec = {
        "type": "json_check",
        "checks": [
            {"path": "sections.*.blocks.*.settings.text", "contains": "Summer Collection 2026"},
            {"path": "sections.*.blocks.*.settings.label", "equals": "Shop Summer"},
        ]
    }
    result = run_verify(spec, files)
    print(f"  L1-01: passed={result['passed']} ({result['passed_checks']}/{result['total_checks']})")
    for f in result["failed_checks"]:
        print(f"    FAIL: {f['reason']}")

    # Test multi_file
    print("\n=== Test multi_file ===")
    files2 = {
        "sections/faq-accordion.liquid": """
<details><summary>Q1</summary><p>A1</p></details>
{% schema %}
{"name": "FAQ Accordion", "blocks": [{"type": "faq-item"}]}
{% endschema %}
""",
        "templates/page.faq.json": json.dumps({
            "sections": {
                "main": {"type": "main-page", "blocks": {}},
                "faq": {"type": "faq-accordion", "blocks": {"q1": {}, "q2": {}, "q3": {}}},
            },
            "order": ["main", "faq"]
        })
    }

    spec2 = {
        "type": "multi_file",
        "checks": {
            "sections/faq-accordion.liquid": [
                {"content_contains": "{% schema %}"},
                {"content_contains": "faq-item"},
                {"content_contains": "question"},
                {"content_contains": "<details"},
            ],
            "templates/page.faq.json": [
                {"json_path": "sections.*.type", "has_value": "faq-accordion"},
                {"json_path": "sections.*.type", "has_value": "main-page"},
                {"json_path": "sections.*.blocks", "min_count": 3},
            ],
        }
    }
    result2 = run_verify(spec2, files2)
    print(f"  L3-01: passed={result2['passed']} ({result2['passed_checks']}/{result2['total_checks']})")
    for f in result2["failed_checks"]:
        print(f"    FAIL: {f['check'].get('file','')}: {f['reason']}")
