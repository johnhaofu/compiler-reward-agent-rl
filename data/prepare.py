"""
Generate training prompts for Horizon theme template generation.

Strategy:
  1. Extract ground truth from existing Horizon templates (13 templates)
  2. Generate variations: different industries, styles, requirements
  3. Generate novel page types not in Horizon
  4. Generate error-prone prompts (edge cases)

Target: 500+ prompts with ground truth templates
"""

import json
import random
from pathlib import Path

# ── Constants ──

INDUSTRIES = [
    "jewelry", "fashion", "electronics", "home decor", "beauty",
    "fitness", "food & beverage", "pet supplies", "toys", "books",
    "outdoor gear", "art & crafts", "automotive", "music", "travel",
    "health supplements", "baby products", "stationery", "gardening",
    "luxury watches",
]

BRAND_STYLES = [
    "minimal & clean", "bold & vibrant", "luxury & elegant",
    "playful & colorful", "rustic & organic", "modern & tech",
    "vintage & retro", "dark & edgy", "warm & cozy", "professional & corporate",
]

PAGE_TYPES = {
    # ── Existing Horizon templates (use as ground truth) ──
    "index": {
        "desc": "Homepage",
        "sections": ["hero", "product-list", "featured-collection", "carousel"],
        "complexity": "high",
    },
    "collection": {
        "desc": "Collection/category page",
        "sections": ["section", "main-collection"],
        "complexity": "medium",
    },
    "product": {
        "desc": "Product detail page",
        "sections": ["product-information", "product-recommendations"],
        "complexity": "high",
    },
    "blog": {
        "desc": "Blog listing page",
        "sections": ["main-blog"],
        "complexity": "low",
    },
    "article": {
        "desc": "Blog article/post page",
        "sections": ["main-blog-post"],
        "complexity": "low",
    },
    "cart": {
        "desc": "Shopping cart page",
        "sections": ["main-cart", "product-list"],
        "complexity": "medium",
    },
    "search": {
        "desc": "Search results page",
        "sections": ["search-header", "search-results"],
        "complexity": "medium",
    },
    "page": {
        "desc": "Generic content page",
        "sections": ["main-page"],
        "complexity": "low",
    },
    "page.contact": {
        "desc": "Contact us page",
        "sections": ["main-page", "section"],
        "complexity": "low",
    },
    "list-collections": {
        "desc": "All collections listing page",
        "sections": ["main-collection-list"],
        "complexity": "medium",
    },
    "password": {
        "desc": "Store password/coming soon page",
        "sections": ["password"],
        "complexity": "low",
    },
    "404": {
        "desc": "404 not found page",
        "sections": ["main-404", "product-list"],
        "complexity": "low",
    },
    # ── Novel page types (no ground truth, model must compose) ──
    "page.about": {
        "desc": "About us page",
        "sections": ["main-page", "hero", "media-with-content"],
        "complexity": "medium",
    },
    "page.faq": {
        "desc": "FAQ page",
        "sections": ["main-page", "section"],
        "complexity": "medium",
    },
    "page.shipping": {
        "desc": "Shipping policy page",
        "sections": ["main-page"],
        "complexity": "low",
    },
    "page.returns": {
        "desc": "Returns & refunds page",
        "sections": ["main-page"],
        "complexity": "low",
    },
    "page.size-guide": {
        "desc": "Size guide page",
        "sections": ["main-page", "section"],
        "complexity": "medium",
    },
    "page.lookbook": {
        "desc": "Lookbook / editorial page",
        "sections": ["hero", "media-with-content", "carousel"],
        "complexity": "high",
    },
    "page.store-locator": {
        "desc": "Store locator page",
        "sections": ["main-page", "section"],
        "complexity": "medium",
    },
    "page.testimonials": {
        "desc": "Customer testimonials page",
        "sections": ["main-page", "section"],
        "complexity": "medium",
    },
}

# ── Prompt templates ──

SYSTEM_PROMPT = """You are an expert Shopify theme developer specializing in the Horizon theme.
Generate valid Shopify template JSON files that are compatible with the Horizon theme.

Rules:
- Output ONLY valid JSON (no comments, no explanation, no markdown)
- The JSON must have "sections" and "order" keys
- Only use section types that exist in the Horizon theme
- Only use block types that exist in the Horizon theme
- Include appropriate settings for each section and block"""

PROMPT_TEMPLATES = [
    # Basic
    "Generate a Shopify Horizon theme template for a {page_desc}.",
    "Create a {page_desc} template for a {industry} store using the Horizon theme.",
    "Build a {page_desc} template with a {brand_style} design aesthetic for Horizon theme.",

    # With specific requirements
    "Generate a {page_desc} template for a {industry} brand. Include: {requirements}.",
    "Create a {brand_style} {page_desc} for a {industry} Shopify store. Must have: {requirements}.",

    # With section hints
    "Generate a {page_desc} template using these sections: {section_hints}. Style: {brand_style}.",
    "Build a {page_desc} for Horizon theme with sections: {section_hints}.",

    # Detailed
    "Create a complete {page_desc} template for a {industry} store. The page should convey a {brand_style} feel. Include the following elements: {requirements}. Use Horizon theme sections and blocks.",

    # Minimal
    "Horizon theme {page_desc} template JSON.",
    "{page_desc} for {industry} store. Horizon theme. {brand_style} style.",
]

REQUIREMENTS_POOL = {
    "index": [
        "hero banner with CTA button",
        "featured products section",
        "product carousel",
        "brand story section",
        "newsletter signup",
        "collection links",
        "social proof section",
        "promotional banner",
    ],
    "collection": [
        "collection title and description",
        "product grid with filters",
        "sorting options",
        "collection image header",
    ],
    "product": [
        "product media gallery",
        "product details with price",
        "add to cart button",
        "product recommendations",
        "size/variant picker",
    ],
    "page": [
        "page title",
        "rich text content",
        "image section",
        "call to action button",
    ],
    "page.contact": [
        "contact form",
        "page title",
        "company information",
        "map or address section",
    ],
    "page.about": [
        "brand story hero",
        "team section",
        "mission statement",
        "company timeline",
        "values section",
    ],
    "page.faq": [
        "accordion for questions",
        "search bar",
        "category tabs",
        "contact link for more help",
    ],
    "default": [
        "heading section",
        "text content",
        "image or media",
        "call to action",
    ],
}


def generate_prompts(count: int = 500, seed: int = 42) -> list[dict]:
    """
    Generate diverse training prompts.

    Returns:
        List of dicts with keys: prompt, template_type, industry, style, system_prompt
    """
    random.seed(seed)
    prompts = []

    # Strategy 1: Direct ground truth prompts (13 templates × ~5 variations = ~65)
    for page_type, info in PAGE_TYPES.items():
        for _ in range(5):
            industry = random.choice(INDUSTRIES)
            style = random.choice(BRAND_STYLES)
            reqs = REQUIREMENTS_POOL.get(page_type, REQUIREMENTS_POOL["default"])
            selected_reqs = random.sample(reqs, min(3, len(reqs)))

            template = random.choice(PROMPT_TEMPLATES)
            prompt = template.format(
                page_desc=info["desc"],
                industry=industry,
                brand_style=style,
                requirements=", ".join(selected_reqs),
                section_hints=", ".join(info["sections"][:3]),
            )

            prompts.append({
                "prompt": prompt,
                "template_type": page_type,
                "industry": industry,
                "style": style,
                "complexity": info["complexity"],
                "system_prompt": SYSTEM_PROMPT,
            })

    # Strategy 2: Industry-focused variations (20 industries × 10 page types = ~200)
    for industry in INDUSTRIES:
        page_types_sample = random.sample(list(PAGE_TYPES.keys()), min(10, len(PAGE_TYPES)))
        for page_type in page_types_sample:
            info = PAGE_TYPES[page_type]
            style = random.choice(BRAND_STYLES)
            reqs = REQUIREMENTS_POOL.get(page_type, REQUIREMENTS_POOL["default"])
            selected_reqs = random.sample(reqs, min(2, len(reqs)))

            template = random.choice(PROMPT_TEMPLATES)
            prompt = template.format(
                page_desc=info["desc"],
                industry=industry,
                brand_style=style,
                requirements=", ".join(selected_reqs),
                section_hints=", ".join(info["sections"][:3]),
            )

            prompts.append({
                "prompt": prompt,
                "template_type": page_type,
                "industry": industry,
                "style": style,
                "complexity": info["complexity"],
                "system_prompt": SYSTEM_PROMPT,
            })

    # Strategy 3: Style-focused variations (~100)
    for style in BRAND_STYLES:
        for _ in range(10):
            page_type = random.choice(list(PAGE_TYPES.keys()))
            info = PAGE_TYPES[page_type]
            industry = random.choice(INDUSTRIES)
            reqs = REQUIREMENTS_POOL.get(page_type, REQUIREMENTS_POOL["default"])
            selected_reqs = random.sample(reqs, min(2, len(reqs)))

            template = random.choice(PROMPT_TEMPLATES)
            prompt = template.format(
                page_desc=info["desc"],
                industry=industry,
                brand_style=style,
                requirements=", ".join(selected_reqs),
                section_hints=", ".join(info["sections"][:3]),
            )

            prompts.append({
                "prompt": prompt,
                "template_type": page_type,
                "industry": industry,
                "style": style,
                "complexity": info["complexity"],
                "system_prompt": SYSTEM_PROMPT,
            })

    # Strategy 4: Complex multi-section prompts (~100)
    for _ in range(100):
        page_type = random.choice(list(PAGE_TYPES.keys()))
        info = PAGE_TYPES[page_type]
        industry = random.choice(INDUSTRIES)
        style = random.choice(BRAND_STYLES)
        reqs = REQUIREMENTS_POOL.get(page_type, REQUIREMENTS_POOL["default"])
        all_reqs = random.sample(reqs, min(4, len(reqs)))

        prompt = (
            f"Generate a complete {info['desc']} template for a {industry} Shopify store "
            f"using the Horizon theme. Design style: {style}. "
            f"The page must include: {', '.join(all_reqs)}. "
            f"Use sections: {', '.join(info['sections'])}. "
            f"Output ONLY valid JSON."
        )

        prompts.append({
            "prompt": prompt,
            "template_type": page_type,
            "industry": industry,
            "style": style,
            "complexity": "high",
            "system_prompt": SYSTEM_PROMPT,
        })

    # Deduplicate by prompt text
    seen = set()
    unique_prompts = []
    for p in prompts:
        if p["prompt"] not in seen:
            seen.add(p["prompt"])
            unique_prompts.append(p)

    # Shuffle and trim to target count
    random.shuffle(unique_prompts)
    final = unique_prompts[:count]

    return final


def load_ground_truths(horizon_path: str) -> dict[str, str]:
    """Load existing Horizon templates as ground truth."""
    gt = {}
    templates_dir = Path(horizon_path) / "templates"
    for f in templates_dir.glob("*.json"):
        content = f.read_text()
        # Strip leading comments
        lines = content.strip().split("\n")
        json_lines = []
        started = False
        for line in lines:
            if not started and line.strip().startswith("{"):
                started = True
            if started:
                json_lines.append(line)
        if json_lines:
            gt[f.stem] = "\n".join(json_lines)
    return gt


def save_dataset(prompts: list[dict], output_path: str, ground_truths: dict = None):
    """Save prompts as JSONL for training."""
    with open(output_path, "w") as f:
        for p in prompts:
            item = {
                "prompt": [
                    {"role": "system", "content": p["system_prompt"]},
                    {"role": "user", "content": p["prompt"]},
                ],
                "template_type": p["template_type"],
                "industry": p["industry"],
                "style": p["style"],
                "complexity": p["complexity"],
            }

            # Add ground truth if available
            if ground_truths and p["template_type"] in ground_truths:
                item["ground_truth"] = ground_truths[p["template_type"]]

            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--horizon-path", type=str, default="/tmp/horizon")
    parser.add_argument("--output", type=str, default="data/prompts/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Generate prompts
    prompts = generate_prompts(count=args.count, seed=args.seed)

    # Load ground truths
    gt = load_ground_truths(args.horizon_path)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_dataset(prompts, args.output, gt)

    # Stats
    print(f"Generated {len(prompts)} prompts → {args.output}")
    print(f"Ground truths available: {len(gt)}")
    print(f"\nTemplate type distribution:")
    from collections import Counter
    type_counts = Counter(p["template_type"] for p in prompts)
    for t, c in type_counts.most_common():
        has_gt = "✅" if t in gt else "❌"
        print(f"  {has_gt} {t}: {c}")
    print(f"\nComplexity distribution:")
    comp_counts = Counter(p["complexity"] for p in prompts)
    for c, n in comp_counts.most_common():
        print(f"  {c}: {n}")
    print(f"\nIndustry coverage: {len(set(p['industry'] for p in prompts))}")
    print(f"Style coverage: {len(set(p['style'] for p in prompts))}")

    # Also generate a small validation set
    val_prompts = generate_prompts(count=50, seed=123)
    val_output = args.output.replace("train", "val")
    save_dataset(val_prompts, val_output, gt)
    print(f"\nValidation set: {len(val_prompts)} prompts → {val_output}")
