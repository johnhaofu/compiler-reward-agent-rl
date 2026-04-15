"""
Generate fixed evaluation set with 3 difficulty levels.

Level 1: Assemble Page (现成组件组装页面)
  - Use existing Horizon sections/blocks to compose template JSON
  - No new .liquid files needed

Level 2: Generate Component (新组件生成)
  - Create new section .liquid files with valid schema
  - Must pass Shopify Liquid syntax + schema validation

Level 3: Component + Page (新组件 + 页面组装)
  - First generate new section(s), then assemble page using them
  - Both component and page must pass validation
"""

import json
from pathlib import Path

SYSTEM_PROMPT = """You are an expert Shopify theme developer specializing in the Horizon theme.
Generate valid Shopify template JSON files and/or Liquid section files that are compatible with the Horizon theme.

Rules:
- Output ONLY valid JSON for templates (no comments, no explanation, no markdown)
- Template JSON must have "sections" and "order" keys
- Only use section types that exist in the Horizon theme OR that you create
- Liquid sections must include a valid {% schema %} block
- Use the tools to research existing patterns before generating"""

# ═══════════════════════════════════════════════════════════════
# Level 1: Assemble Page — 用现成组件组装页面
# ═══════════════════════════════════════════════════════════════

LEVEL_1_ASSEMBLE = [
    # ── 简单页面 (1-2 sections) ──
    {"id": "L1-01", "template_type": "page", "industry": "books",
     "prompt": "Generate a simple content page template using the main-page section with a heading and page content blocks.",
     "expected_sections": ["main-page"]},

    {"id": "L1-02", "template_type": "page.contact", "industry": "jewelry",
     "prompt": "Generate a contact us page using main-page section for title/content and a section with contact-form block.",
     "expected_sections": ["main-page", "section"]},

    {"id": "L1-03", "template_type": "password", "industry": "fashion",
     "prompt": "Generate a store password/coming soon page using the password section with logo, text, and email signup blocks.",
     "expected_sections": ["password"]},

    {"id": "L1-04", "template_type": "404", "industry": "electronics",
     "prompt": "Generate a 404 page using main-404 section with error text and a button, plus a product-list section for recommendations.",
     "expected_sections": ["main-404", "product-list"]},

    {"id": "L1-05", "template_type": "blog", "industry": "food",
     "prompt": "Generate a blog listing page using the main-blog section with title and blog post card blocks.",
     "expected_sections": ["main-blog"]},

    {"id": "L1-06", "template_type": "article", "industry": "travel",
     "prompt": "Generate a blog article page using main-blog-post section with post title, details, image, and content blocks.",
     "expected_sections": ["main-blog-post"]},

    # ── 中等页面 (2-3 sections) ──
    {"id": "L1-07", "template_type": "collection", "industry": "fashion",
     "prompt": "Generate a collection page with a section for collection title/description and main-collection for product grid with filters.",
     "expected_sections": ["section", "main-collection"]},

    {"id": "L1-08", "template_type": "search", "industry": "beauty",
     "prompt": "Generate a search results page using search-header section for the search bar and search-results section for filterable product grid.",
     "expected_sections": ["search-header", "search-results"]},

    {"id": "L1-09", "template_type": "cart", "industry": "toys",
     "prompt": "Generate a cart page using main-cart section with cart title, items, and summary blocks, plus a product-list for recommendations.",
     "expected_sections": ["main-cart", "product-list"]},

    {"id": "L1-10", "template_type": "list-collections", "industry": "gardening",
     "prompt": "Generate an all-collections page using main-collection-list section with collection cards.",
     "expected_sections": ["main-collection-list"]},

    # ── 复杂页面 (3+ sections) ──
    {"id": "L1-11", "template_type": "index", "industry": "jewelry",
     "prompt": "Generate a homepage using hero section with text and button blocks, followed by a product-list section for featured products.",
     "expected_sections": ["hero", "product-list"]},

    {"id": "L1-12", "template_type": "index", "industry": "fitness",
     "prompt": "Generate a homepage with hero section, carousel section for featured items, and product-list section for best sellers.",
     "expected_sections": ["hero", "carousel", "product-list"]},

    {"id": "L1-13", "template_type": "product", "industry": "watches",
     "prompt": "Generate a product detail page using product-information section with media gallery and product details blocks, plus product-recommendations section.",
     "expected_sections": ["product-information", "product-recommendations"]},

    {"id": "L1-14", "template_type": "product", "industry": "beauty",
     "prompt": "Generate a product page with product-information section, a media-with-content section for product story, and product-recommendations.",
     "expected_sections": ["product-information", "media-with-content", "product-recommendations"]},
]

# ═══════════════════════════════════════════════════════════════
# Level 2: Generate Component — 新组件生成
# ═══════════════════════════════════════════════════════════════

LEVEL_2_COMPONENT = [
    # ── 简单组件 (static content) ──
    {"id": "L2-01", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'brand-story.liquid' that displays a brand story with a heading, paragraph text, and an optional background image. Include a proper {% schema %} with settings for heading text, body text, and image picker.",
     "output_file": "sections/brand-story.liquid",
     "industry": "jewelry"},

    {"id": "L2-02", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'announcement-banner.liquid' for displaying a promotional announcement. Include schema settings for announcement text, background color, text color, and a link.",
     "output_file": "sections/announcement-banner.liquid",
     "industry": "fashion"},

    {"id": "L2-03", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'trust-badges.liquid' showing trust/guarantee badges in a horizontal row. Include schema settings for up to 4 badges, each with an icon image, title, and description.",
     "output_file": "sections/trust-badges.liquid",
     "industry": "electronics"},

    # ── 中等组件 (with blocks) ──
    {"id": "L2-04", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'faq-accordion.liquid' with collapsible FAQ items. Use blocks of type 'faq-item' with settings for question and answer text. Include JavaScript for toggle behavior.",
     "output_file": "sections/faq-accordion.liquid",
     "industry": "general"},

    {"id": "L2-05", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'testimonial-grid.liquid' displaying customer testimonials in a grid. Use blocks of type 'testimonial' with settings for customer name, quote text, rating (1-5), and optional avatar image.",
     "output_file": "sections/testimonial-grid.liquid",
     "industry": "beauty"},

    {"id": "L2-06", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'team-members.liquid' showing team member cards. Use blocks of type 'member' with settings for name, role, photo, and bio text. Display in a responsive grid.",
     "output_file": "sections/team-members.liquid",
     "industry": "general"},

    # ── 复杂组件 (with Liquid logic) ──
    {"id": "L2-07", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'collection-tabs.liquid' that shows products from multiple collections in a tabbed interface. Include schema settings for selecting up to 3 collections, and use Liquid to loop through collection products.",
     "output_file": "sections/collection-tabs.liquid",
     "industry": "fashion"},

    {"id": "L2-08", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'countdown-timer.liquid' for a sale countdown. Include schema settings for end date, heading text, and colors. Use JavaScript for the countdown logic.",
     "output_file": "sections/countdown-timer.liquid",
     "industry": "electronics"},

    {"id": "L2-09", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'instagram-feed.liquid' that displays an instagram-style image grid. Use blocks of type 'image-post' with settings for image, caption, and link. Include hover effects with CSS.",
     "output_file": "sections/instagram-feed.liquid",
     "industry": "beauty"},

    {"id": "L2-10", "template_type": "section",
     "prompt": "Generate a new Liquid section called 'size-guide-table.liquid' with a responsive size chart table. Use blocks of type 'size-row' with settings for size label and measurements. Include schema presets.",
     "output_file": "sections/size-guide-table.liquid",
     "industry": "fashion"},
]

# ═══════════════════════════════════════════════════════════════
# Level 3: Component + Page — 新组件 + 页面组装
# ═══════════════════════════════════════════════════════════════

LEVEL_3_FULL = [
    {"id": "L3-01", "template_type": "page.about", "industry": "jewelry",
     "prompt": "Create an About Us page for a jewelry brand. First generate a 'brand-story.liquid' section with heading, story text, and image. Then generate a 'team-members.liquid' section with member blocks. Finally, assemble a page.about.json template using main-page for title, your brand-story section, and your team-members section.",
     "output_files": ["sections/brand-story.liquid", "sections/team-members.liquid", "templates/page.about.json"]},

    {"id": "L3-02", "template_type": "page.faq", "industry": "electronics",
     "prompt": "Create a FAQ page for an electronics store. First generate a 'faq-accordion.liquid' section with collapsible question-answer blocks. Then assemble a page.faq.json template using main-page for the page title and your faq-accordion section for the questions.",
     "output_files": ["sections/faq-accordion.liquid", "templates/page.faq.json"]},

    {"id": "L3-03", "template_type": "page.testimonials", "industry": "beauty",
     "prompt": "Create a testimonials page for a beauty brand. First generate a 'testimonial-grid.liquid' section with customer review blocks (name, quote, rating). Then assemble a page.testimonials.json template using main-page for title and your testimonial-grid section.",
     "output_files": ["sections/testimonial-grid.liquid", "templates/page.testimonials.json"]},

    {"id": "L3-04", "template_type": "page.lookbook", "industry": "fashion",
     "prompt": "Create a lookbook page for a fashion brand. First generate an 'instagram-feed.liquid' section with image-post blocks. Then assemble a page.lookbook.json template using hero section for the cover, media-with-content for editorial text, and your instagram-feed section for the image grid.",
     "output_files": ["sections/instagram-feed.liquid", "templates/page.lookbook.json"]},

    {"id": "L3-05", "template_type": "page.size-guide", "industry": "fashion",
     "prompt": "Create a size guide page. First generate a 'size-guide-table.liquid' section with a responsive measurement table using blocks for each size row. Then assemble a page.size-guide.json template using main-page for title/intro and your size-guide-table section.",
     "output_files": ["sections/size-guide-table.liquid", "templates/page.size-guide.json"]},

    {"id": "L3-06", "template_type": "page.shipping", "industry": "food",
     "prompt": "Create a shipping policy page. First generate a 'trust-badges.liquid' section with guarantee/shipping badges. Then assemble a page.shipping.json template using main-page for the policy content and your trust-badges section at the top.",
     "output_files": ["sections/trust-badges.liquid", "templates/page.shipping.json"]},

    {"id": "L3-07", "template_type": "index", "industry": "fitness",
     "prompt": "Create an enhanced homepage for a fitness store. First generate a 'countdown-timer.liquid' section for a sale promotion. Then assemble an index.json template with hero section, your countdown-timer, product-list for featured products, and featured-blog-posts for tips.",
     "output_files": ["sections/countdown-timer.liquid", "templates/index.json"]},

    {"id": "L3-08", "template_type": "collection", "industry": "electronics",
     "prompt": "Create an enhanced collection page. First generate a 'collection-tabs.liquid' section that shows products from multiple collections in tabs. Then assemble a collection.json template with section for collection header, your collection-tabs, and main-collection for the full grid.",
     "output_files": ["sections/collection-tabs.liquid", "templates/collection.json"]},
]


def main():
    output_path = Path("data/prompts/eval_fixed.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = []

    # Level 1
    for item in LEVEL_1_ASSEMBLE:
        all_items.append({
            "id": item["id"],
            "level": 1,
            "level_name": "assemble_page",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ],
            "template_type": item["template_type"],
            "industry": item["industry"],
            "complexity": "low" if len(item.get("expected_sections", [])) <= 2 else "medium",
            "expected_sections": item.get("expected_sections", []),
        })

    # Level 2
    for item in LEVEL_2_COMPONENT:
        all_items.append({
            "id": item["id"],
            "level": 2,
            "level_name": "generate_component",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ],
            "template_type": item["template_type"],
            "industry": item["industry"],
            "complexity": "medium",
            "output_file": item.get("output_file", ""),
        })

    # Level 3
    for item in LEVEL_3_FULL:
        all_items.append({
            "id": item["id"],
            "level": 3,
            "level_name": "component_and_page",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ],
            "template_type": item["template_type"],
            "industry": item["industry"],
            "complexity": "high",
            "output_files": item.get("output_files", []),
        })

    with open(output_path, "w") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    print(f"Fixed eval set: {len(all_items)} samples → {output_path}")
    print(f"\nLevel breakdown:")
    print(f"  Level 1 (Assemble Page):      {len(LEVEL_1_ASSEMBLE)} tasks")
    print(f"  Level 2 (Generate Component): {len(LEVEL_2_COMPONENT)} tasks")
    print(f"  Level 3 (Component + Page):   {len(LEVEL_3_FULL)} tasks")
    print(f"  Total:                         {len(all_items)} tasks")

    print(f"\nLevel 1 page types: {sorted(set(i['template_type'] for i in LEVEL_1_ASSEMBLE))}")
    print(f"Level 2 components: {[i['output_file'] for i in LEVEL_2_COMPONENT]}")
    print(f"Level 3 deliverables: {[i['output_files'] for i in LEVEL_3_FULL]}")


if __name__ == "__main__":
    main()
