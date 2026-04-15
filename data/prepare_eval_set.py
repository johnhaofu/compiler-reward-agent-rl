"""
Fixed evaluation set based on real Horizon theme customization tasks.

Each task:
  - Starts from an existing Horizon template (real baseline)
  - Has a specific customization requirement
  - Has a programmatic verify function checking the output
  - Can be visually verified via Playwright screenshot

3 levels:
  L1: Modify existing template (change settings, add/remove blocks)
  L2: Add new section to existing template
  L3: Create new section .liquid + add to template

Preview URL: https://sitemusestore.myshopify.com/?preview_theme_id=185572393248
"""

import json
from pathlib import Path


SYSTEM_PROMPT = """You are an expert Shopify theme developer working with the Horizon theme.
Your task is to customize the theme by modifying template JSON files and/or creating new Liquid section files.

Rules:
- Output valid JSON for templates (must have "sections" and "order" keys)
- Only use section types that exist in the Horizon theme or that you create
- Liquid sections must include a valid {% schema %} block
- Use the tools to research existing patterns before making changes
- Always validate your changes before calling done"""

PREVIEW_BASE = "https://sitemusestore.myshopify.com"
THEME_ID = "185572393248"


def _check_json(data):
    """Helper: ensure basic JSON structure."""
    return isinstance(data, dict) and "sections" in data and "order" in data


# ═══════════════════════════════════════════════════════════
# Level 1: Modify existing template
# ═══════════════════════════════════════════════════════════

LEVEL_1 = [
    {
        "id": "L1-01",
        "name": "Change hero heading text",
        "base_template": "index",
        "prompt": "Modify the homepage template: change the hero section heading text to '<p>Summer Collection 2026</p>' and the button label to 'Shop Summer'.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.blocks.*.settings.text", "contains": "Summer Collection 2026"},
                {"path": "sections.*.blocks.*.settings.label", "equals": "Shop Summer"},
            ]
        },
    },
    {
        "id": "L1-02",
        "name": "Change product card size to large",
        "base_template": "collection",
        "prompt": "Modify the collection template: change the main-collection section's product_card_size setting to 'large' for bigger product cards on desktop.",
        "preview_path": "/collections/all",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.main.settings.product_card_size", "equals": "large"},
            ]
        },
    },
    {
        "id": "L1-03",
        "name": "Add a second text block to hero",
        "base_template": "index",
        "prompt": "Modify the homepage template: add a new text block to the hero section with the content '<p>Free shipping on orders over $50</p>'. Place it after the existing heading text block.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.blocks.*.settings.text", "contains": "Free shipping"},
                {"path": "sections.*.type", "has_value": "hero"},
            ]
        },
    },
    {
        "id": "L1-04",
        "name": "Change hero color scheme",
        "base_template": "index",
        "prompt": "Modify the homepage template: change the hero section's color_scheme setting to 'scheme-2' and set toggle_overlay to false.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.settings.color_scheme", "equals": "scheme-2"},
                {"path": "sections.*.settings.toggle_overlay", "equals": False},
            ]
        },
    },
    {
        "id": "L1-05",
        "name": "Change product list to show 8 products",
        "base_template": "index",
        "prompt": "Modify the homepage template: change the product-list section to display 8 products maximum by setting max_products to 8.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.settings.max_products", "equals": 8},
            ]
        },
    },
    {
        "id": "L1-06",
        "name": "Remove product recommendations from product page",
        "base_template": "product",
        "prompt": "Modify the product template: remove the product-recommendations section entirely. Keep only the main product-information section.",
        "preview_path": "/products/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "order", "length_equals": 1},
                {"path": "sections.*.type", "has_value": "product-information"},
                {"path": "sections.*.type", "not_has_value": "product-recommendations"},
            ]
        },
    },
    {
        "id": "L1-07",
        "name": "Add email signup block to password page",
        "base_template": "password",
        "prompt": "Modify the password template: ensure it has a logo block, a text block with content 'Coming Soon - Launching Spring 2026', and an email-signup block.",
        "preview_path": "/password",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.blocks.*.type", "has_value": "logo"},
                {"path": "sections.*.blocks.*.type", "has_value": "email-signup"},
                {"path": "sections.*.blocks.*.settings.text", "contains": "Coming Soon"},
            ]
        },
    },
    {
        "id": "L1-08",
        "name": "Change contact page form to centered",
        "base_template": "page.contact",
        "prompt": "Modify the contact page template: change the form section's horizontal_alignment_flex_direction_column setting to 'center' so the contact form is centered on the page.",
        "preview_path": "/pages/contact",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.form.settings.horizontal_alignment_flex_direction_column", "equals": "center"},
            ]
        },
    },
]

# ═══════════════════════════════════════════════════════════
# Level 2: Add existing section to template
# ═══════════════════════════════════════════════════════════

LEVEL_2 = [
    {
        "id": "L2-01",
        "name": "Add marquee section to homepage",
        "base_template": "index",
        "prompt": "Modify the homepage template: add a marquee section between the hero and the product list. The marquee should have a text block with content 'FREE SHIPPING ON ALL ORDERS • NEW ARRIVALS WEEKLY •'. Set the speed setting to 'medium'.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.type", "has_value": "marquee"},
                {"path": "sections.*.blocks.*.settings.text", "contains": "FREE SHIPPING"},
                {"path": "order", "value_before": ["marquee", "product"]},
            ]
        },
    },
    {
        "id": "L2-02",
        "name": "Add media-with-content section to homepage",
        "base_template": "index",
        "prompt": "Add a media-with-content section after the product list on the homepage. Include a text block with heading 'Our Story' and body text 'We believe in quality ingredients for better sleep.'. Set section_height to 'medium'.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.type", "has_value": "media-with-content"},
                {"path": "sections.*.blocks.*.settings.text", "contains": "Our Story"},
                {"path": "sections.*.settings.section_height", "equals": "medium"},
            ]
        },
    },
    {
        "id": "L2-03",
        "name": "Add carousel to homepage",
        "base_template": "index",
        "prompt": "Add a carousel section after the hero on the homepage. Add 3 card blocks to the carousel, each with a text block. The cards should have text 'Best Sellers', 'New Arrivals', and 'On Sale'.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.type", "has_value": "carousel"},
                {"path": "order", "value_before": ["hero", "carousel"]},
                {"path": "sections.*.blocks", "min_count": 3},
            ]
        },
    },
    {
        "id": "L2-04",
        "name": "Add divider between sections on collection page",
        "base_template": "collection",
        "prompt": "Add a divider section between the collection header section and the main-collection section on the collection page. Use the divider section type.",
        "preview_path": "/collections/all",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.type", "has_value": "divider"},
                {"path": "order", "length_gte": 3},
            ]
        },
    },
    {
        "id": "L2-05",
        "name": "Add product-list recommendations to 404 page",
        "base_template": "404",
        "prompt": "Modify the 404 page: ensure there is a product-list section showing recommended products. Set max_products to 4 and columns to 4. Add a header block with text 'You might also like'.",
        "preview_path": "/404",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.type", "has_value": "product-list"},
                {"path": "sections.*.settings.max_products", "equals": 4},
                {"path": "sections.*.settings.columns", "equals": 4},
            ]
        },
    },
    {
        "id": "L2-06",
        "name": "Add featured-blog-posts to homepage",
        "base_template": "index",
        "prompt": "Add a featured-blog-posts section at the end of the homepage (before footer). It should display the latest blog posts.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.type", "has_value": "featured-blog-posts"},
                {"path": "order", "last_contains": "blog"},
            ]
        },
    },
    {
        "id": "L2-07",
        "name": "Add slideshow to homepage replacing hero",
        "base_template": "index",
        "prompt": "Replace the hero section on the homepage with a slideshow section. Add 2 slide blocks, each with text content. First slide: 'Summer Sale - Up to 50% Off', second slide: 'New Arrivals - Shop Now'. Keep the product-list section.",
        "preview_path": "/",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.type", "has_value": "slideshow"},
                {"path": "sections.*.type", "not_has_value": "hero"},
                {"path": "sections.*.blocks", "min_count": 2},
            ]
        },
    },
    {
        "id": "L2-08",
        "name": "Build a multi-section about page",
        "base_template": "page",
        "prompt": "Create a page.about.json template with: 1) A hero section with heading 'About Us' and a button 'Learn More', 2) A media-with-content section with heading 'Our Mission' and text 'We create products that help you sleep better.', 3) A main-page section for additional page content.",
        "preview_path": "/pages/about",
        "verify": {
            "type": "json_check",
            "checks": [
                {"path": "sections.*.type", "has_value": "hero"},
                {"path": "sections.*.type", "has_value": "media-with-content"},
                {"path": "sections.*.type", "has_value": "main-page"},
                {"path": "sections.*.blocks.*.settings.text", "contains": "About Us"},
                {"path": "sections.*.blocks.*.settings.text", "contains": "Our Mission"},
                {"path": "order", "length_gte": 3},
            ]
        },
    },
]

# ═══════════════════════════════════════════════════════════
# Level 3: Create new section + add to template
# ═══════════════════════════════════════════════════════════

LEVEL_3 = [
    {
        "id": "L3-01",
        "name": "Create FAQ accordion section + FAQ page",
        "base_template": "page",
        "prompt": "Create a new section file 'sections/faq-accordion.liquid' with: 1) An accordion section using <details>/<summary> HTML elements, 2) Block type 'faq-item' with 'question' and 'answer' text settings in the schema, 3) Proper {% schema %} with name 'FAQ Accordion'. Then create a page.faq.json template that uses main-page section for the page title and your faq-accordion section with 3 faq-item blocks about shipping, returns, and sizing.",
        "preview_path": "/pages/faq",
        "output_files": ["sections/faq-accordion.liquid", "templates/page.faq.json"],
        "verify": {
            "type": "multi_file",
            "checks": {
                "sections/faq-accordion.liquid": [
                    {"content_contains": "{% schema %}"},
                    {"content_contains": "faq-item"},
                    {"content_contains": "question"},
                    {"content_contains": "answer"},
                    {"content_contains": "<details"},
                ],
                "templates/page.faq.json": [
                    {"json_path": "sections.*.type", "has_value": "faq-accordion"},
                    {"json_path": "sections.*.type", "has_value": "main-page"},
                    {"json_path": "sections.*.blocks", "min_count": 3},
                ],
            }
        },
    },
    {
        "id": "L3-02",
        "name": "Create testimonials section + testimonials page",
        "base_template": "page",
        "prompt": "Create a new section 'sections/testimonial-cards.liquid' with: 1) A grid of testimonial cards, 2) Block type 'testimonial' with settings: customer_name (text), quote (textarea), rating (range 1-5), 3) Schema with name 'Testimonials'. Then create page.testimonials.json using main-page for title and your testimonial-cards section with 3 testimonial blocks.",
        "preview_path": "/pages/testimonials",
        "output_files": ["sections/testimonial-cards.liquid", "templates/page.testimonials.json"],
        "verify": {
            "type": "multi_file",
            "checks": {
                "sections/testimonial-cards.liquid": [
                    {"content_contains": "{% schema %}"},
                    {"content_contains": "testimonial"},
                    {"content_contains": "customer_name"},
                    {"content_contains": "quote"},
                    {"content_contains": "rating"},
                ],
                "templates/page.testimonials.json": [
                    {"json_path": "sections.*.type", "has_value": "testimonial-cards"},
                    {"json_path": "sections.*.blocks", "min_count": 3},
                ],
            }
        },
    },
    {
        "id": "L3-03",
        "name": "Create countdown timer + add to homepage",
        "base_template": "index",
        "prompt": "Create a new section 'sections/countdown-banner.liquid' with: 1) A countdown timer display using JavaScript, 2) Schema settings: end_date (text), heading (text), background_color (color), text_color (color), 3) Name 'Countdown Banner'. Then modify the homepage index.json to add this countdown-banner section between the hero and product list. Set heading to 'Summer Sale Ends In' and end_date to '2026-08-01'.",
        "preview_path": "/",
        "output_files": ["sections/countdown-banner.liquid", "templates/index.json"],
        "verify": {
            "type": "multi_file",
            "checks": {
                "sections/countdown-banner.liquid": [
                    {"content_contains": "{% schema %}"},
                    {"content_contains": "end_date"},
                    {"content_contains": "countdown"},
                ],
                "templates/index.json": [
                    {"json_path": "sections.*.type", "has_value": "countdown-banner"},
                    {"json_path": "sections.*.type", "has_value": "hero"},
                    {"json_path": "order", "value_before": ["hero", "countdown"]},
                ],
            }
        },
    },
    {
        "id": "L3-04",
        "name": "Create size guide table + product page integration",
        "base_template": "product",
        "prompt": "Create a new section 'sections/size-guide.liquid' with: 1) A responsive HTML table for size measurements, 2) Block type 'size-row' with settings: size_label (text), chest (text), waist (text), hips (text), 3) Schema with name 'Size Guide'. Then modify product.json to add the size-guide section after product-information.",
        "preview_path": "/products/",
        "output_files": ["sections/size-guide.liquid", "templates/product.json"],
        "verify": {
            "type": "multi_file",
            "checks": {
                "sections/size-guide.liquid": [
                    {"content_contains": "{% schema %}"},
                    {"content_contains": "size-row"},
                    {"content_contains": "<table"},
                    {"content_contains": "chest"},
                ],
                "templates/product.json": [
                    {"json_path": "sections.*.type", "has_value": "size-guide"},
                    {"json_path": "sections.*.type", "has_value": "product-information"},
                ],
            }
        },
    },
    {
        "id": "L3-05",
        "name": "Create newsletter popup section + add to homepage",
        "base_template": "index",
        "prompt": "Create a new section 'sections/newsletter-popup.liquid' with: 1) A modal/popup overlay for email signup, 2) Schema settings: heading (text), description (textarea), button_label (text), delay_seconds (range 0-30), 3) JavaScript to show popup after delay. Then add it to the homepage index.json. Set heading to 'Get 10% Off', description to 'Subscribe to our newsletter for exclusive deals.', delay to 5 seconds.",
        "preview_path": "/",
        "output_files": ["sections/newsletter-popup.liquid", "templates/index.json"],
        "verify": {
            "type": "multi_file",
            "checks": {
                "sections/newsletter-popup.liquid": [
                    {"content_contains": "{% schema %}"},
                    {"content_contains": "delay"},
                    {"content_contains": "newsletter"},
                ],
                "templates/index.json": [
                    {"json_path": "sections.*.type", "has_value": "newsletter-popup"},
                    {"json_path": "sections.*.type", "has_value": "hero"},
                ],
            }
        },
    },
    {
        "id": "L3-06",
        "name": "Create brand values section + about page",
        "base_template": "page",
        "prompt": "Create a new section 'sections/brand-values.liquid' with: 1) A 3-column grid of value cards with icons, 2) Block type 'value-card' with settings: icon (image_picker), title (text), description (textarea), 3) Schema with name 'Brand Values'. Then create page.about.json with hero section (heading 'Our Story'), your brand-values section with 3 cards (Quality, Sustainability, Community), and main-page section.",
        "preview_path": "/pages/about",
        "output_files": ["sections/brand-values.liquid", "templates/page.about.json"],
        "verify": {
            "type": "multi_file",
            "checks": {
                "sections/brand-values.liquid": [
                    {"content_contains": "{% schema %}"},
                    {"content_contains": "value-card"},
                    {"content_contains": "icon"},
                    {"content_contains": "title"},
                ],
                "templates/page.about.json": [
                    {"json_path": "sections.*.type", "has_value": "brand-values"},
                    {"json_path": "sections.*.type", "has_value": "hero"},
                    {"json_path": "sections.*.blocks.*.settings.text", "contains": "Our Story"},
                    {"json_path": "order", "length_gte": 3},
                ],
            }
        },
    },
]


def main():
    output_path = Path("data/prompts/eval_fixed.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = []
    for item in LEVEL_1:
        all_items.append({**item, "level": 1, "level_name": "modify_template",
                          "system_prompt": SYSTEM_PROMPT, "preview_url": f"{PREVIEW_BASE}{item['preview_path']}?preview_theme_id={THEME_ID}"})
    for item in LEVEL_2:
        all_items.append({**item, "level": 2, "level_name": "add_section",
                          "system_prompt": SYSTEM_PROMPT, "preview_url": f"{PREVIEW_BASE}{item['preview_path']}?preview_theme_id={THEME_ID}"})
    for item in LEVEL_3:
        all_items.append({**item, "level": 3, "level_name": "create_component",
                          "system_prompt": SYSTEM_PROMPT, "preview_url": f"{PREVIEW_BASE}{item['preview_path']}?preview_theme_id={THEME_ID}"})

    # Write as JSONL
    with open(output_path, "w") as f:
        for item in all_items:
            record = {
                "id": item["id"],
                "level": item["level"],
                "level_name": item["level_name"],
                "name": item["name"],
                "prompt": [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["prompt"]},
                ],
                "base_template": item.get("base_template", ""),
                "preview_url": item.get("preview_url", ""),
                "verify": item["verify"],
                "output_files": item.get("output_files", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Fixed eval set: {len(all_items)} tasks → {output_path}")
    print(f"\n  L1 Modify Template:    {len(LEVEL_1)} tasks")
    print(f"  L2 Add Section:        {len(LEVEL_2)} tasks")
    print(f"  L3 Create Component:   {len(LEVEL_3)} tasks")
    print(f"  Total:                 {len(all_items)} tasks")

    print(f"\nL1 tasks:")
    for t in LEVEL_1:
        print(f"  {t['id']}: {t['name']}")
    print(f"\nL2 tasks:")
    for t in LEVEL_2:
        print(f"  {t['id']}: {t['name']}")
    print(f"\nL3 tasks:")
    for t in LEVEL_3:
        print(f"  {t['id']}: {t['name']}")


if __name__ == "__main__":
    main()
