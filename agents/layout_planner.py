from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tools.testing_decorator import measure_time
from tools.style_planner import StylePlanner, StylePlan
from tools.rag_retriever import DesignKnowledgeBase, DesignSnippet
from tools.layout_utils import BUILTIN_TEMPLATES, LayoutTemplate, PosterComposer


class LayoutPlannerAgent:
    """
    Layout Planner (Final Poster Version)
    - Use RAG + StylePlanner to generate layout, color scheme, and font plan
    - Integrate Visual Designer’s main image and brief copy into a complete poster
    - Output PNG (with text/decorations) and pass to Quality Refiner for scoring
    """
    def __init__(
            self,
            output_dir: str = "../outputs/posters",
            knowledge_base: DesignKnowledgeBase | None = None,
    ) -> None:
        # Resolve relative paths based on project root
        base_path = Path(__file__).parent.parent
        self.output_dir = (base_path / output_dir.lstrip("../")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.knowledge_base = knowledge_base or DesignKnowledgeBase()
        self.templates = BUILTIN_TEMPLATES
        self.composer = PosterComposer()
        self.style_planner = StylePlanner()

    @measure_time("LayoutPlanner")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        brief = state.get("expanded_concept", {})
        visual_output = state.get("visual_output", "")
        if len(brief) == 0:
            raise ValueError("Concept expander is empty")
        if len(visual_output) == 0:
            raise ValueError("Visual output is empty")

        if isinstance(brief, str):
            brief = brief.strip()
            if brief.startswith("```"):
                lines = brief.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                brief = '\n'.join(lines).strip()

            try:
                import json
                brief = json.loads(brief)
                print("   ✅ Parsed successfully")
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parsing failed: {e}")
                print(f"   Content: {brief[:200]}...")
                raise ValueError(
                    f"Failed to parse expanded_concept: {e}\n"
                    f"Content: {brief[:200]}..."
                )

        subject_path = Path(visual_output.get("image_path"))
        if not subject_path.exists():
            raise FileNotFoundError(f"Visual Designer image doesn't exist：{subject_path}")

        snippets = self._query_design_rules(brief)
        style_plan = self.style_planner.plan(brief, snippets)
        template = self.templates[style_plan.template_key]
        copywriting = self._prepare_copywriting(brief, style_plan)
        text_blocks = self._build_text_blocks(template, style_plan, copywriting)
        decor_context = self._build_decor_context(style_plan)

        poster_path = self._compose_poster(subject_path, template, style_plan.palette, text_blocks, decor_context)

        return {
            "layout_output": {"layout_path": poster_path}
        }

    # Perform RAG and make decisions
    def _query_design_rules(self, brief: Dict[str, Any]) -> List[DesignSnippet]:
        keywords = [
            brief.get("theme", ""),
            brief.get("style", ""),
            brief.get("tone", ""),
            brief.get("emotion", ""),
            brief.get("color_palette", ""),
        ]
        query = " ".join(filter(None, keywords)) or "poster layout best practices"
        return self.knowledge_base.retrieve(query, top_k=3)

    # Rendering
    def _compose_poster(
            self,
            subject_path: Path,
            template: LayoutTemplate,
            palette: Dict[str, str],
            text_blocks: List[Dict[str, Any]],
            decor_context: Dict[str, Any],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"poster_{timestamp}.png"
        output_path = self.output_dir / filename
        self.composer.compose(
            subject_image_path=subject_path,
            template=template,
            palette=palette,
            text_blocks=text_blocks,
            decor_context=decor_context,
            output_path=output_path,
        )
        return output_path

    # Copywriting & typography helpers
    def _prepare_copywriting(self, brief: Dict[str, Any], style_plan: StylePlan) -> Dict[str, Any]:
        title = (
                brief.get("title")
                or brief.get("theme")
                or brief.get("product_name")
                or "Gourmet Delight"
        )
        subtitle = brief.get("tagline") or brief.get("emotion") or "Artisan Flavor"
        description = brief.get("description") or style_plan.storyline
        selling_points = brief.get("selling_points", [])
        if isinstance(selling_points, str):
            selling_points = [pt.strip() for pt in selling_points.replace("。", ".").split(".") if pt.strip()]
        if not selling_points and description:
            selling_points = self._split_into_phrases(description)
        bullet_points = selling_points[:3]
        body_text = description if not bullet_points else ""
        cta = brief.get("cta") or "立即品尝"
        badge = brief.get("badge") or style_plan.badge_text

        # Keep only 3–4 words in the body text of all templates
        if body_text:
            words = body_text.split()
            if len(words) > 4:
                body_text = " ".join(words[:4])

        # Add new field
        discount = brief.get("discount", "")
        # Use short features to ensure single-line display (2–3 words) by default
        default_features = ["ORGANIC FRESH", "HEALTHY CHOICE", "HOT & SPICY"]
        features = brief.get("features", default_features)
        if isinstance(features, str):
            features = [f.strip() for f in features.split(",")]

        # All templates must use three feature lines with bullet points
        # Strictly limit each line to 2–3 words to ensure no line wrapping
        short_features = []
        for f in features[:3]:
            words = f.split()
            # Keep only the first 2-3 words
            if len(words) > 3:
                short_features.append(" ".join(words[:2]))
            else:
                short_features.append(f)

        # If the features list has fewer than 3 items, use default values to fill in
        while len(short_features) < 3:
            short_features.append(default_features[len(short_features)])

        features_text = "\n".join(f"• {f}" for f in short_features[:3])

        contact = brief.get("contact", "PHONE: 012-666999")
        hours = brief.get("hours", "OPEN 7AM - 9PM")

        return {
            "title": title,
            "subtitle": subtitle,
            "body": body_text,
            "body_items": bullet_points if bullet_points else None,
            "cta": cta,
            "badge": badge,
            "discount": discount,
            "features": features_text,
            "contact": contact,
            "hours": hours,
        }

    def _split_into_phrases(self, text: str) -> List[str]:
        if not text:
            return []
        for sep in ["。", "！", "？", ".", ";", "，"]:
            text = text.replace(sep, "|")
        phrases = [phrase.strip() for phrase in text.split("|") if phrase.strip()]
        return phrases

    def _build_text_blocks(
            self,
            template: LayoutTemplate,
            style_plan: StylePlan,
            copywriting: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        palette = style_plan.palette
        typography = style_plan.typography
        body_block_info = None
        title_block_info = None

        # First pass: build all blocks and record title and body block information
        for region in template.text_regions:
            # Skip the subtitle area; do not render subtitles anymore
            if region.role == "subtitle":
                continue

            text_value = copywriting.get(region.role)
            items_value = copywriting.get(f"{region.role}_items")

            # Ensure the CTA contains text
            if region.role == "cta":
                if not text_value or (isinstance(text_value, str) and not text_value.strip()):
                    text_value = copywriting.get("cta") or "Taste Immediately"

            if not text_value and not items_value:
                continue

            style = typography.get(region.role, typography.get("body", {}))

            # Force body, title, and feature blocks to be center-aligned
            block_align = region.align
            if region.role in ("body", "title", "features"):
                block_align = "center"

            block = {
                "text": text_value,
                "items": items_value,
                "role": region.role,
                "x": region.x,
                "y": region.y,
                "max_width": region.max_width,
                "font_size": int(region.font_size * style.get("size_scale", 1.0)),
                "align": block_align,
                "line_spacing": region.line_spacing or int(region.font_size * 1.2),
                "weight": style.get("weight", region.weight),
                "font_family": style.get("family", "sans_clean"),
                "color": style.get("color") or palette.get("text_primary", "#1C1C1C"),
                "uppercase": style.get("uppercase", region.uppercase),
            }

            # For specific roles, override the font family to ensure cursive fonts are used
            if region.role == "features":
                block["font_family"] = "script"

            if items_value and style.get("bullet", True):
                block["bullet"] = style.get("bullet_prefix", "• ")

            if style.get("bg_color"):
                block["bg_color"] = style["bg_color"]
                block["padding"] = style.get("padding", 18)
                block["border_radius"] = style.get("border_radius", 30 if region.role != "badge" else 999)

            if region.role == "cta":
                block.setdefault("bg_color", style.get("bg_color", palette.get("accent")))
                block.setdefault("color", style.get("color", "#FFFFFF"))
                block.setdefault("padding", style.get("padding", 28))
                block.setdefault("border_radius", style.get("border_radius", 60))
                block["text_shadow"] = True
                block["shadow_alpha"] = 120
                # Ensure CTA not null
                if not block["text"]:
                    block["text"] = "Taste Immediately!"

            if region.role == "badge":
                block.setdefault("bg_color", style.get("bg_color", palette.get("background_alt", "#FFFFFF")))
                block.setdefault("color", style.get("color", palette.get("accent", "#C8744B")))
                block.setdefault("padding", style.get("padding", 18))
                block.setdefault("border_radius", style.get("border_radius", 999))
                block["uppercase"] = True
                block["border_color"] = palette.get("accent", "#C8744B")
                block["border_width"] = 2
                block["border_alpha"] = 180
                block["text_shadow"] = True
                block["shadow_color"] = "#000000"
                block["shadow_alpha"] = 60
                block["shadow_offset_x"] = 2
                block["shadow_offset_y"] = 2

            if region.role == "title":
                block["force_single_line"] = True
                title_block_info = {
                    "block": block,
                    "region": region,
                    "style": style,
                }

            if region.role == "body":
                body_block_info = {
                    "block": block,
                    "region": region,
                    "style": style,
                }
            blocks.append(block)

        # Second pass: if both body and CTA exist, dynamically adjust the CTA’s y-coordinate
        if body_block_info:
            body_block = body_block_info["block"]
            body_region = body_block_info["region"]

            # Estimate the actual number of lines and height of the body
            body_text = body_block.get("text", "")
            body_items = body_block.get("items", [])
            body_lines = []
            if body_items:
                bullet = body_block.get("bullet", "• ")
                for item in body_items:
                    if item and item.strip():
                        body_lines.append(f"{bullet}{item}")
            elif body_text and body_text.strip():
                # Simple estimation: each line contains about max_width / (font_size * 0.6) characters
                font_size = body_block["font_size"]
                max_width = body_block["max_width"]
                chars_per_line = max(1, int(max_width / (font_size * 0.6)))
                body_lines = [body_text[i:i + chars_per_line] for i in range(0, len(body_text), chars_per_line)]

            if body_lines:
                line_spacing = body_block["line_spacing"]
                body_height = len(body_lines) * line_spacing
                body_padding = body_block.get("padding", 18)
                body_bottom = body_region.y + body_height + (body_padding * 2)

                for block in blocks:
                    if block.get("role") == "cta":
                        min_spacing = 180
                        new_cta_y = body_bottom + min_spacing
                        canvas_height = template.canvas_size[1]
                        cta_padding = block.get("padding", 28)
                        cta_font_size = block.get("font_size", 44)
                        if new_cta_y + cta_font_size + (cta_padding * 2) < canvas_height - 40:
                            block["y"] = new_cta_y
                        break
        return blocks

    def _build_decor_context(self, style_plan: StylePlan) -> Dict[str, Any]:
        return {
            "food_category": style_plan.food_category,
            "mood": style_plan.mood,
            "palette": style_plan.palette,
            "extra_decor": style_plan.extra_decor,
            "motif_density": 8 if style_plan.food_category in ("dessert", "beverage") else 5,
            "shape_count": 15,
            "icon_count": 6,
            "qr_content": "https://your-restaurant.com/menu",
            "ribbon_y": 160,
        }