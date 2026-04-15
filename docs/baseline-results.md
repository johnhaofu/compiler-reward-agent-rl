# Baseline Experiment Results

## Experiment Settings

### Evaluation Benchmark
- **Tasks**: 22 fixed eval tasks on Shopify Horizon theme customization
  - L1 (8 tasks): Modify existing templates (change settings, add/remove blocks)
  - L2 (8 tasks): Add existing sections to templates (marquee, carousel, slideshow, etc.)
  - L3 (6 tasks): Create new .liquid sections + assemble pages (FAQ, testimonials, countdown, etc.)
- **Verification**: API validation (Sitemuse themeFilesUpsert) + programmatic verify checks (JSON path checks, content checks)
- **Max turns**: 50
- **Tools**: 9 tools — list_files, read_file, grep, write_file, edit_json, list_components, get_section_schema, validate, done

### Models

| Setting | Claude Sonnet 4 | Qwen3.5-4B |
|---------|-----------------|------------|
| Model ID | claude-sonnet-4-20250514 | Qwen/Qwen3.5-4B |
| Parameters | ~200B (estimated) | 4B (hybrid GatedDeltaNet+MoE) |
| Architecture | Dense Transformer | 8 x (3x GatedDeltaNet + 1x Attention) |
| Inference | Anthropic API | vLLM (local, RTX 4080 32GB) |
| Context | 200K | 262K |
| Temperature | 1.0 | 0.6 |
| Max tokens/turn | 4096 | 8192 |
| Thinking mode | N/A | Enabled (reasoning_content stripped from history) |
| Cost | $0.51/task avg | $0 (local GPU) |

### Infrastructure
- **GPU Server**: AutoDL RTX 4080 32GB, CUDA 12.x
- **vLLM**: Standard release, FlashAttention v2, Triton/FLA GDN kernel
- **Tool parser**: `--tool-call-parser qwen3_coder --reasoning-parser qwen3`

---

## Overall Results

| Metric | Claude Sonnet 4 | Qwen3.5-4B | Delta |
|--------|:-:|:-:|:-:|
| **Fully Resolved** | **18/22 (81.8%)** | **12/22 (54.5%)** | **-27.3%** |
| API Resolved | 20/22 (90.9%) | 15/22 (68.2%) | -22.7% |
| First-try Valid | 15/22 (68.2%) | 8/22 (36.4%) | -31.8% |
| Fix Rate | 5/22 (22.7%) | 7/22 (31.8%) | +9.1% |
| Verify Passed | 18/22 (81.8%) | 13/22 (59.1%) | -22.7% |

---

## Results by Difficulty Level

### L1: Modify Template (8 tasks)

| Metric | Claude | Qwen3.5 |
|--------|:-:|:-:|
| Fully Resolved | 7/8 (87.5%) | 7/8 (87.5%) |
| First-try Valid | 7/8 (87.5%) | 6/8 (75.0%) |
| Fix Rate | 0/8 (0%) | 1/8 (12.5%) |
| Avg Turns | 7.8 | 13.5 |
| Avg Tokens | 47,823 | 181,792 |
| Avg Time | 41.8s | 66.5s |
| Total Errors | 1 | 1 |

### L2: Add Section (8 tasks)

| Metric | Claude | Qwen3.5 |
|--------|:-:|:-:|
| Fully Resolved | 6/8 (75.0%) | 4/8 (50.0%) |
| API Resolved | 8/8 (100%) | 6/8 (75.0%) |
| First-try Valid | 4/8 (50.0%) | 1/8 (12.5%) |
| Fix Rate | 4/8 (50.0%) | 5/8 (62.5%) |
| Avg Turns | 15.4 | 29.6 |
| Avg Tokens | 191,524 | 697,341 |
| Avg Time | 88.0s | 248.1s |
| Total Errors | 10 | 19 |

### L3: Create Component (6 tasks)

| Metric | Claude | Qwen3.5 |
|--------|:-:|:-:|
| Fully Resolved | 5/6 (83.3%) | 1/6 (16.7%) |
| API Resolved | 5/6 (83.3%) | 2/6 (33.3%) |
| First-try Valid | 4/6 (66.7%) | 1/6 (16.7%) |
| Fix Rate | 1/6 (16.7%) | 1/6 (16.7%) |
| Avg Turns | 17.7 | 38.3 |
| Avg Tokens | 269,412 | 1,014,269 |
| Avg Time | 105.8s | 305.7s |
| Total Errors | 1 | 83 |

---

## Efficiency Metrics

| Metric | Claude | Qwen3.5 | Ratio |
|--------|:-:|:-:|:-:|
| Avg Turns | 13.2 | 26.1 | 2.0x |
| Avg Tokens | 160,511 | 596,303 | 3.7x |
| Total Tokens | 3,531,243 | 13,118,673 | 3.7x |
| Avg Input Tokens | 157,881 | 587,644 | 3.7x |
| Avg Output Tokens | 2,630 | 8,660 | 3.3x |
| Avg Time | 76.0s | 197.8s | 2.6x |
| Avg Validations | 1.6 | 2.6 | 1.6x |
| Avg Write Calls | 0.5 | 2.0 | 4.0x |
| Avg Research Turns | 7.2 | 12.8 | 1.8x |
| Avg Fix Turns | 0.8 | 2.1 | 2.6x |
| Total Tool Calls | 291 | 575 | 2.0x |
| Total Cost | $11.29 | $0.00 | - |

---

## Tool Usage

| Tool | Claude | Qwen3.5 | Ratio |
|------|:-:|:-:|:-:|
| read_file | 59 | 167 | 2.8x |
| grep | 99 | 165 | 1.7x |
| validate | 35 | 58 | 1.7x |
| edit_json | 31 | 49 | 1.6x |
| write_file | 11 | 43 | 3.9x |
| list_files | 8 | 37 | 4.6x |
| get_section_schema | 17 | 24 | 1.4x |
| list_components | 10 | 17 | 1.7x |
| done | 21 | 15 | 0.7x |
| **Total** | **291** | **575** | **2.0x** |

---

## Error Analysis

| Error Type | Claude | Qwen3.5 |
|------------|:-:|:-:|
| other | 10 | 75 |
| invalid_block_type | 2 | 13 |
| liquid_syntax | 0 | 6 |
| order_mismatch | 0 | 4 |
| unknown_key | 0 | 3 |
| missing_key | 0 | 2 |
| **Total Errors** | **12** | **103** |
| **Unique Errors** | **11** | **62** |

---

## Per-Task Results

| Task | Level | Claude | Qwen3.5 | Q-FTV | Q-Fix | Q-Turns | C-Turns | Q-Tokens |
|------|:-----:|:------:|:-------:|:-----:|:-----:|:-------:|:-------:|:--------:|
| L1-01 | L1 | ✅ | ✅ | ✓ | - | 5 | 4 | 17,590 |
| L1-02 | L1 | ✅ | ✅ | ✓ | - | 6 | 10 | 23,925 |
| L1-03 | L1 | ✅ | ✅ | ✓ | - | 11 | 6 | 108,641 |
| L1-04 | L1 | ✅ | ✅ | ✓ | - | 5 | 4 | 17,032 |
| L1-05 | L1 | **❌** | **✅** | ✓ | - | 23 | 11 | 380,046 |
| L1-06 | L1 | ✅ | ❌ | - | - | 42 | 18 | 777,010 |
| L1-07 | L1 | ✅ | ✅ | ✓ | - | 8 | 4 | 47,339 |
| L1-08 | L1 | ✅ | ✅ | - | ✓ | 8 | 5 | 82,751 |
| L2-01 | L2 | ✅ | ✅ | - | ✓ | 48 | 13 | 1,221,312 |
| L2-02 | L2 | ✅ | ❌ | - | - | 29 | 26 | 532,598 |
| L2-03 | L2 | ❌ | ❌ | - | ✓* | 52 | 17 | 2,028,773 |
| L2-04 | L2 | ✅ | ✅ | ✓ | - | 8 | 7 | 49,460 |
| L2-05 | L2 | ✅ | ✅ | - | ✓ | 27 | 6 | 501,454 |
| L2-06 | L2 | ✅ | ❌ | - | - | 38 | 27 | 659,220 |
| L2-07 | L2 | ✅ | ✅ | - | ✓ | 16 | 9 | 258,439 |
| L2-08 | L2 | ❌ | ❌ | - | ✓* | 19 | 18 | 327,472 |
| L3-01 | L3 | ❌ | ❌ | ✓* | - | 14 | 1 | 61,741 |
| L3-02 | L3 | ✅ | ❌ | - | - | 52 | 10 | 1,422,132 |
| L3-03 | L3 | ✅ | ❌ | - | - | 23 | 14 | 429,737 |
| L3-04 | L3 | ✅ | ❌ | - | - | 50 | 41 | 1,545,507 |
| L3-05 | L3 | ✅ | ❌† | - | - | 50 | 20 | 1,541,081 |
| L3-06 | L3 | ✅ | ✅ | - | ✓ | 41 | 20 | 1,085,413 |

\* API resolved but verify failed
† API failed due to edit_json string-as-operations bug, but verify passed

---

## Key Observations

### Qwen3.5-4B Strengths
1. **L1 parity**: 87.5% matches Claude's 87.5% on template modification tasks
2. **L1-05 outperformed Claude**: Qwen solved a task Claude failed on
3. **Higher fix rate** (31.8% vs 22.7%): More willing to retry after errors
4. **Zero cost**: Local GPU inference vs $11.29 for Claude API

### Qwen3.5-4B Weaknesses
1. **L3 collapse** (16.7% vs 83.3%): Cannot compose new .liquid + JSON template together
2. **Repetitive reads**: 2.8x more read_file calls — model forgets file content between turns
3. **grep loops**: Gets stuck trying regex patterns to find JSON structures in multi-line files
4. **edit_json format errors**: Passes operations as JSON string instead of array (tool bug, fixable)
5. **Token inefficiency**: 3.7x more tokens per task due to repeated exploration
6. **83 errors on L3** vs Claude's 1: Liquid syntax errors, invalid block types, missing keys

### RL Training Opportunities
1. **Reduce repetitive reads**: Train model to retain file context across turns
2. **Fix tool argument format**: Correct edit_json string-as-operations pattern
3. **Improve L3 component creation**: Biggest gap (17% → 83%), highest reward potential
4. **Shorter trajectories**: Average 26 turns → target 15 (Claude-level efficiency)
5. **Compiler feedback utilization**: 103 validation errors provide rich learning signals

---

## Reproduction

### Claude Sonnet 4 Baseline
```bash
python experiments/eval_claude_baseline.py \
  --data-path data/prompts/eval_fixed.jsonl \
  --max-samples 22 --max-turns 50
```

### Qwen3.5-4B Baseline
```bash
# Start vLLM server
vllm serve /root/autodl-tmp/models/Qwen3.5-4B \
  --port 8000 --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder

# Run evaluation
python experiments/eval_qwen_baseline.py \
  --api-base http://localhost:8000/v1 \
  --horizon-path /root/autodl-tmp/horizon \
  --max-samples 22 --max-turns 50 \
  --output-path experiments/results/qwen3.5_baseline_run1.json
```

### Result Files
- `experiments/results/claude_baseline_v1.json` — Claude Sonnet 4 (★18/22)
- `experiments/results/qwen3.5_baseline_run1.json` — Qwen3.5-4B (★12/22)
