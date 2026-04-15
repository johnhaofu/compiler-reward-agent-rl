"""
Prepare SFT training data from ground_truth prompts.

Synthesizes expert agent trajectories:
  list_components → get_section_schema → write_file(ground_truth) → validate(pass) → done

Usage:
  python training/prepare_sft_data.py
  python training/prepare_sft_data.py --include-research  # add list_files/read_file exploration steps
"""

import json
import random
import hashlib
from pathlib import Path

SYSTEM_PROMPT = """You are an expert Shopify theme developer working with the Horizon theme.

## Your Task
Generate valid Shopify theme template files based on the user's requirements.
The templates must be compatible with the Horizon theme and pass Shopify's validation.

## Available Tools
- **list_files**: Browse theme directories to see available sections, blocks, templates
- **read_file**: Read existing files to understand patterns and schemas
- **grep**: Search across files for specific patterns or settings usage
- **write_file**: Create or update files in the workspace
- **edit_json**: Apply JSON Patch operations to modify existing JSON files (more efficient than rewriting)
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
- IMPORTANT: Some sections have static blocks (like 'static-header', 'static-product-card') that are auto-managed by the section. Do NOT include static blocks in your template JSON — only include non-static blocks you want to customize. If you see static blocks when reading an existing template, omit them when writing your version.
"""

# Template type → file path mapping
TEMPLATE_PATHS = {
    "index": "templates/index.json",
    "product": "templates/product.json",
    "collection": "templates/collection.json",
    "blog": "templates/blog.json",
    "article": "templates/article.json",
    "cart": "templates/cart.json",
    "search": "templates/search.json",
    "404": "templates/404.json",
    "password": "templates/password.json",
    "page": "templates/page.json",
    "page.contact": "templates/page.contact.json",
    "page.faq": "templates/page.faq.json",
    "list-collections": "templates/list-collections.json",
}

# Pre-loaded component list summary (from data/horizon_components.json)
COMPONENTS_SUMMARY = None


def load_components_summary() -> str:
    """Load pre-extracted component list."""
    comp_path = Path("data/horizon_components.json")
    if comp_path.exists():
        data = json.loads(comp_path.read_text())
        sections = data.get("sections", {})
        # sections is a dict: {name: {display_name, settings_count, block_types, ...}}
        section_list = [
            {"name": name, "display_name": info.get("display_name", name)}
            for name, info in list(sections.items())[:20]
        ]
        return json.dumps({
            "sections": section_list,
            "total_sections": len(sections)
        }, indent=2)[:2000]
    return '{"sections": [], "note": "component list unavailable"}'


def load_schema(section_name: str) -> str:
    """Load schema for a section type."""
    schema_path = Path(f"data/schemas/{section_name}.json")
    if schema_path.exists():
        return schema_path.read_text()[:3000]
    return json.dumps({"error": f"Schema not found for {section_name}"})


def extract_section_types(ground_truth: str) -> list[str]:
    """Extract section type names from ground_truth JSON."""
    try:
        data = json.loads(ground_truth)
        sections = data.get("sections", {})
        types = set()
        for sec in sections.values():
            if isinstance(sec, dict) and "type" in sec:
                types.add(sec["type"])
        return list(types)
    except (json.JSONDecodeError, AttributeError):
        return []


def make_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call-{hashlib.md5(str(random.random()).encode()).hexdigest()[:12]}"


def synthesize_trajectory(
    prompt: dict,
    include_research: bool = True,
) -> list[dict] | None:
    """Synthesize an expert trajectory from a prompt with ground_truth.

    Returns a list of messages in OpenAI chat format, or None if not possible.
    """
    ground_truth = prompt.get("ground_truth")
    if not ground_truth:
        return None

    template_type = prompt.get("template_type", "page")
    file_path = TEMPLATE_PATHS.get(template_type, f"templates/{template_type}.json")
    user_content = prompt["prompt"][-1]["content"]

    # Extract section types used in ground_truth
    section_types = extract_section_types(ground_truth)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Step 1: Research — list_components
    if include_research:
        tc_id = make_tool_call_id()
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "list_components",
                    "arguments": {}
                }
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": load_components_summary(),
        })

    # Step 2: Research — get_section_schema for main section type
    if include_research and section_types:
        main_type = section_types[0]
        schema_content = load_schema(main_type)
        tc_id = make_tool_call_id()
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "get_section_schema",
                    "arguments": {"section_name": main_type}
                }
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": schema_content,
        })

    # Step 3: Write file
    tc_id = make_tool_call_id()
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": {
                    "path": file_path,
                    "content": ground_truth
                }
            }
        }]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc_id,
        "content": json.dumps({
            "success": True,
            "path": file_path,
            "size": len(ground_truth)
        }),
    })

    # Step 4: Validate
    tc_id = make_tool_call_id()
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {
                "name": "validate",
                "arguments": {"files": [file_path]}
            }
        }]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc_id,
        "content": json.dumps({
            "passed": True,
            "files_validated": [file_path]
        }),
    })

    # Step 5: Done
    tc_id = make_tool_call_id()
    summary = f"Created {file_path} with {len(section_types)} sections. Validation passed."
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {
                "name": "done",
                "arguments": {
                    "status": "success",
                    "summary": summary
                }
            }
        }]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc_id,
        "content": json.dumps({
            "status": "success",
            "summary": summary
        }),
    })

    return messages


def prepare_sft_dataset(
    train_path: str = "data/prompts/train.jsonl",
    output_path: str = "data/sft/train.jsonl",
    include_research: bool = True,
    seed: int = 42,
):
    """Prepare SFT training dataset from ground_truth prompts."""
    random.seed(seed)

    # Load training prompts
    prompts = []
    with open(train_path) as f:
        for line in f:
            prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} training prompts")
    gt_prompts = [p for p in prompts if p.get("ground_truth")]
    print(f"  {len(gt_prompts)} have ground_truth")

    # Synthesize trajectories
    trajectories = []
    for p in gt_prompts:
        traj = synthesize_trajectory(p, include_research=include_research)
        if traj:
            trajectories.append({
                "messages": traj,
                "template_type": p.get("template_type", ""),
                "complexity": p.get("complexity", ""),
                "industry": p.get("industry", ""),
                "source": "synthetic",
            })

    print(f"Synthesized {len(trajectories)} trajectories")

    # Shuffle
    random.shuffle(trajectories)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")

    # Stats
    avg_msgs = sum(len(t["messages"]) for t in trajectories) / len(trajectories)
    avg_turns = sum(
        sum(1 for m in t["messages"] if m["role"] == "assistant")
        for t in trajectories
    ) / len(trajectories)
    print(f"  Avg messages/trajectory: {avg_msgs:.1f}")
    print(f"  Avg assistant turns: {avg_turns:.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/prompts/train.jsonl")
    parser.add_argument("--output-path", default="data/sft/train.jsonl")
    parser.add_argument("--include-research", action="store_true", default=True)
    parser.add_argument("--no-research", dest="include_research", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_sft_dataset(
        train_path=args.train_path,
        output_path=args.output_path,
        include_research=args.include_research,
        seed=args.seed,
    )
