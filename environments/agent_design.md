# Coding Agent Design: Tools & System Prompt

## Task Definition

**Input**: Natural language page requirement (e.g. "Generate an About Us page for a jewelry brand")
**Output**: Valid Horizon theme template files (JSON + optionally Liquid sections)
**Environment**: Shopify Horizon theme directory (read-only base + writable workspace)
**Validation**: Sitemuse API upsert (Shopify themeFilesUpsert GraphQL)

---

## Tool Definitions

### 1. list_files

```json
{
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
```

**Returns**: List of filenames
**Example**: `list_files("sections/")` → `["hero.liquid", "main-page.liquid", "carousel.liquid", ...]`
**Next-state signal**: Low information (just a list). Low entropy expected.

---

### 2. read_file

```json
{
  "name": "read_file",
  "description": "Read the contents of a file in the Horizon theme. Use to study existing templates, section schemas, and code patterns.",
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
```

**Returns**: File content as string
**Example**: `read_file("templates/page.json")` → `{"sections": {"main": ...}, "order": ["main"]}`
**Next-state signal**: Rich (full file content). Medium entropy — agent needs to decide what to do with the info.

---

### 3. grep

```json
{
  "name": "grep",
  "description": "Search for a pattern across files in the theme. Use to find how specific settings, block types, or patterns are used.",
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
```

**Returns**: Matching lines with file paths
**Example**: `grep("color_scheme", "sections/")` → `sections/hero.liquid:42: "color_scheme": "scheme-6"`
**Next-state signal**: Medium. Helps agent understand patterns.

---

### 4. write_file

```json
{
  "name": "write_file",
  "description": "Write content to a file in the workspace. Use to create or update template JSON files and section Liquid files. Only writes to the workspace directory, not the base theme.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Relative path for the file. Examples: 'templates/page.about.json', 'sections/custom-about.liquid'"
      },
      "content": {
        "type": "string",
        "description": "The full file content to write"
      }
    },
    "required": ["path", "content"]
  }
}
```

**Returns**: `{"success": true, "path": "...", "size": 1234}` or `{"error": "..."}`
**Next-state signal**: Low (just confirmation). The real signal comes from validate.

---

### 5. validate

```json
{
  "name": "validate",
  "description": "Validate theme files by upserting to Shopify via API. Returns detailed error messages if validation fails. Use after writing template files to check correctness.",
  "parameters": {
    "type": "object",
    "properties": {
      "files": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Relative file path. Examples: 'templates/page.about.json', 'sections/custom-hero.liquid'"
            }
          }
        },
        "description": "List of files to validate. Can validate multiple files at once."
      }
    },
    "required": ["files"]
  }
}
```

**Returns**:
```json
// Success
{"passed": true, "files_validated": ["templates/page.about.json"]}

// Failure with actionable errors
{
  "passed": false,
  "errors": [
    {"file": "templates/page.about.json", "message": "Section type 'about-hero' does not refer to an existing section file"},
    {"file": "templates/page.about.json", "message": "Section id 'ghost_section' must exist in sections"}
  ]
}
```

**Next-state signal**: **HIGHEST VALUE**. Error messages are precise, actionable hints.
This is the core signal for:
- Compiler-OPD (error message → hint → teacher distribution)
- Error-Branch (failure → fork multiple fix strategies)
- Process Reward (pass/fail at each validate call)

---

### 6. done

```json
{
  "name": "done",
  "description": "Signal that the task is complete. Call this when validation passes or you've exhausted your fix attempts.",
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
```

---

## System Prompt

```
You are an expert Shopify theme developer working with the Horizon theme.

## Your Task
Generate valid Shopify theme template files based on the user's requirements.
The templates must be compatible with the Horizon theme and pass Shopify's validation.

## Available Tools
You have access to these tools:
- list_files: Browse theme directories to see available sections, blocks, templates
- read_file: Read existing files to understand patterns and schemas
- grep: Search across files for specific patterns or settings usage
- write_file: Create or update files in the workspace
- validate: Check files against Shopify's theme validation API
- done: Signal task completion

## Workflow
Follow this workflow to create high-quality templates:

1. **Research**: Use list_files and read_file to understand the Horizon theme structure.
   Study existing templates for the same or similar page types.
   Check section schemas to understand required settings.

2. **Plan**: Decide which sections and blocks to use for the requested page.
   Consider the page type, industry context, and design requirements.

3. **Generate**: Use write_file to create the template JSON file.
   Follow the Horizon theme conventions you observed in step 1.

4. **Validate**: Use validate to check the generated files.
   If validation fails, read the error messages carefully.

5. **Fix**: If there are errors, fix them based on the error messages.
   Use read_file to re-examine reference files if needed.
   Use write_file to update the fixed content.
   Use validate again.

6. **Complete**: Call done when validation passes.

## Rules
- Template JSON must have "sections" and "order" keys
- Only use section types that exist in the Horizon theme (check with list_files)
- Only use block types that exist in the Horizon theme (check with list_files)
- Settings must match the section/block schema (check with read_file on .liquid files)
- Output valid JSON only in write_file — no comments, no markdown
- Always validate before calling done
- If validation fails, read the error message and fix the specific issue
- Maximum {max_turns} tool calls allowed
```

---

## Tool Call Format

Using Qwen3-Coder's native tool calling format:

```
<tool_call>
{"name": "list_files", "arguments": {"path": "sections/"}}
</tool_call>
```

Tool response:
```
<tool_response>
["hero.liquid", "main-page.liquid", "carousel.liquid", ...]
</tool_response>
```

---

## Example Episode

```
User: Generate a Contact Us page for a jewelry store with the Horizon theme.

Agent thinks: I need to create a contact page. Let me first check what exists.

<tool_call>
{"name": "list_files", "arguments": {"path": "templates/"}}
</tool_call>

<tool_response>
["404.json", "article.json", "blog.json", "cart.json", "collection.json",
 "index.json", "list-collections.json", "page.contact.json", "page.json",
 "password.json", "product.json", "search.json"]
</tool_response>

Agent thinks: There's already a page.contact.json. Let me read it as reference.

<tool_call>
{"name": "read_file", "arguments": {"path": "templates/page.contact.json"}}
</tool_call>

<tool_response>
{"sections": {"main": {"type": "main-page", "blocks": {"title": {"type": "text", ...}, "content": {"type": "text", ...}}, ...}, "form": {"type": "section", "blocks": {"contact_form_UwiCkQ": {"type": "contact-form", ...}}, ...}}, "order": ["main", "form"]}
</tool_response>

Agent thinks: I see the pattern. main-page for title/content, section with contact-form block.
Let me customize it for a jewelry store.

<tool_call>
{"name": "write_file", "arguments": {"path": "templates/page.contact-jewelry.json", "content": "{\"sections\": {\"main\": {\"type\": \"main-page\", ...}, \"form\": {\"type\": \"section\", ...}}, \"order\": [\"main\", \"form\"]}"}}
</tool_call>

<tool_response>
{"success": true, "path": "templates/page.contact-jewelry.json", "size": 892}
</tool_response>

<tool_call>
{"name": "validate", "arguments": {"files": [{"path": "templates/page.contact-jewelry.json"}]}}
</tool_call>

<tool_response>
{"passed": true, "files_validated": ["templates/page.contact-jewelry.json"]}
</tool_response>

<tool_call>
{"name": "done", "arguments": {"status": "success", "summary": "Created contact page template for jewelry store with title, content, and contact form sections."}}
</tool_call>
```

---

## Metrics Captured Per Episode

| Metric | Description | Use |
|--------|-------------|-----|
| pass@1 | Passed on first validate call | Zero-shot capability |
| pass@N | Passed within N validate calls | Fix capability |
| total_turns | Total tool calls made | Efficiency |
| validate_calls | Number of validate calls | Fix attempts |
| research_turns | list_files + read_file + grep calls before first write | Planning quality |
| fix_turns | Tool calls after first failed validate | Fix efficiency |
| error_types | Types of errors encountered | Error analysis |
| tool_call_sequence | Ordered list of tools used | Behavior analysis |

---

## RL Training Signal Sources

| Tool | Signal Type | For Which Method |
|------|------------|-----------------|
| validate (fail) | Binary reward (0/1) | Outcome Reward (B2) |
| validate (error msg) | OPD hint | Compiler-OPD (M1) |
| validate (fail) | Branch trigger | Error-Branch (M2) |
| validate (each call) | Process reward | Process Reward |
| read_file (return) | Next-state context | ARPO entropy analysis |
| grep (return) | Next-state context | ARPO entropy analysis |

validate is the most valuable tool — it simultaneously provides:
1. Binary reward signal
2. Precise error messages for OPD
3. Branch trigger for exploration
4. Process reward at each invocation
