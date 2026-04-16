"""
tools/style_planner.py
---------------------------------
Generate layout, color scheme, typography, and decoration plans
based on the brief and RAG results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class StylePlan:
    template_key: str
    palette: Dict[str, str]
    typography: Dict[str, Dict[str, Any]]
    food_category: str
    mood: str
    badge_text: str
    extra_decor: List[str]
    storyline: str

class StylePlanner:
    def __init__(self) -> None:
        self.typography_presets: Dict[str, Dict[str, Dict[str, Any]]] = {
            "luxury": {
                "title": {"family": "serif_elegant", "weight": "bold", "color": "#F6E5C9", "uppercase": True},
                "subtitle": {"family": "sans_clean", "color": "#E6C28F"},
                "body": {"family": "sans_clean", "color": "#F1DDC2"},
                "cta": {"family": "sans_bold", "color": "#1B1410", "bg_color": "#F9C97B"},
                "badge": {"family": "sans_bold", "color": "#1B1410", "bg_color": "#F6E5C9"},
            },
            "warm_dessert": {
                "title": {"family": "serif_elegant", "weight": "bold", "color": "#3B1C12"},
                "subtitle": {"family": "script", "color": "#C0623B"},
                "body": {"family": "sans_clean", "color": "#4B2B20"},
                "cta": {"family": "sans_bold", "color": "#FFFFFF", "bg_color": "#C8744B"},
                "badge": {"family": "sans_bold", "color": "#C8744B", "bg_color": "#FFFFFF"},
            },
            "fresh_drink": {
                "title": {"family": "sans_bold", "weight": "bold", "color": "#0E2A36", "uppercase": True},
                "subtitle": {"family": "sans_clean", "color": "#1D5064"},
                "body": {"family": "sans_clean", "color": "#103544"},
                "cta": {"family": "sans_bold", "color": "#FFFFFF", "bg_color": "#3A8FB7"},
                "badge": {"family": "sans_bold", "color": "#0E2A36", "bg_color": "#9BD3E2"},
            },
            "coffee": {
                "title": {"family": "serif_elegant", "weight": "bold", "color": "#2B1D14"},
                "subtitle": {"family": "script", "color": "#8B5A3C"},
                "body": {"family": "sans_clean", "color": "#3B2A21"},
                "cta": {"family": "sans_bold", "color": "#FDF0DA", "bg_color": "#7C4F32"},
                "badge": {"family": "sans_bold", "color": "#7C4F32", "bg_color": "#FDF0DA"},
            },
        }

    def plan(self, brief: Dict[str, Any], snippets: List[Any]) -> StylePlan:
        category = self._infer_food_category(brief, snippets)
        template_key = self._select_template_key(brief, snippets, category)
        palette = self._select_palette(brief, snippets, category)
        mood = brief.get("tone", "warm").lower()
        preset_key = self._select_typography_preset(brief, category, palette)
        typography = self._build_typography(preset_key)
        badge_text = self._build_badge_text(brief, category)

        extra_decor: List[str] = []

        storyline = brief.get("story", "") or brief.get("description", "") or brief.get("emotion", "delightful")

        return StylePlan(
            template_key=template_key,
            palette=palette,
            typography=typography,
            food_category=category,
            mood=mood,
            badge_text=badge_text,
            extra_decor=list(set(extra_decor)),  # 去重
            storyline=storyline,
        )

    # Heuristics
    def _infer_food_category(self, brief: Dict[str, Any], snippets: List[Any]) -> str:
        keywords = (brief.get("theme", "") + " " + brief.get("description", "") + " ").lower()
        keywords += " ".join(getattr(s, "text", "") for s in snippets).lower()
        mapping = {
            "dessert": ["dessert", "cake", "cream", "sweet", "mousse", "tart"],
            "bakery": ["bread", "pastry", "croissant", "bagel"],
            "coffee": ["coffee", "latte", "espresso"],
            "tea": ["tea", "matcha", "milk tea", "oolong"],
            "beverage": ["juice", "smoothie", "drink", "soda", "mocktail"],
            "savory": ["noodle", "soup", "burger", "pizza", "ramen", "steak"],
        }
        for category, tokens in mapping.items():
            if any(token in keywords for token in tokens):
                return category
        return "generic"

    def _select_template_key(self, brief: Dict[str, Any], snippets: List[Any], category: str) -> str:
        text = (brief.get("style", "") + " " + brief.get("tone", "")).lower()
        text += " ".join(getattr(s, "text", "") for s in snippets).lower()

        if brief.get("promotional") or brief.get("discount") or "sale" in text or "promotion" in text:
            return "grid_info_rich"

        if any(word in text for word in ["vibrant", "youthful", "energetic", "活力", "年轻"]):
            return "vibrant_collage"

        if category in ("beverage", "juice") or any(word in text for word in ["modern", "fresh", "brunch"]):
            return "diagonal_split"

        if any(word in text for word in ["luxury", "premium", "night", "dark", "奢华", "黑金"]):
            return "spotlight_dark"

        return "soft_glow_portrait"

    def _select_palette(self, brief: Dict[str, Any], snippets: List[Any], category: str) -> Dict[str, str]:
        text = (brief.get("tone", "") + " " + brief.get("style", "")).lower()
        text += " ".join(getattr(s, "text", "") for s in snippets).lower()
        if any(word in text for word in ["luxury", "premium", "night", "dark", "black", "金"]):
            return {
                "background": "#18121E",
                "background_alt": "#33201D",
                "accent": "#E9C07D",
                "accent_alt": "#F8E2B7",
                "text_primary": "#F6E7CC",
            }
        if category in ("dessert", "bakery"):
            return {
                "background": "#F9E5D3",
                "background_alt": "#FAD6C3",
                "accent": "#C8744B",
                "accent_alt": "#FFE1C5",
                "text_primary": "#3C1F13",
            }
        if category in ("beverage", "juice"):
            return {
                "background": "#E3F1F8",
                "background_alt": "#C7DCEB",
                "accent": "#3A8FB7",
                "accent_alt": "#8ED1E3",
                "text_primary": "#0E2A36",
            }
        if category in ("coffee", "tea"):
            return {
                "background": "#F4EFE7",
                "background_alt": "#DACBB7",
                "accent": "#7C4F32",
                "accent_alt": "#E6D2BE",
                "text_primary": "#2B1D14",
            }
        return {
            "background": "#EFEFEF",
            "background_alt": "#DBDBDB",
            "accent": "#2F80ED",
            "accent_alt": "#9BC1FF",
            "text_primary": "#1C1C1C",
        }

    def _select_typography_preset(self, brief: Dict[str, Any], category: str, palette: Dict[str, str]) -> str:
        text = (brief.get("style", "") + " " + brief.get("tone", "")).lower()
        if any(word in text for word in ["luxury", "premium", "黑金", "复古"]) or palette["background"] == "#18121E":
            return "luxury"
        if category in ("dessert", "bakery"):
            return "warm_dessert"
        if category in ("coffee", "tea"):
            return "coffee"
        if category in ("beverage", "juice"):
            return "fresh_drink"
        return "warm_dessert"

    def _build_typography(self, preset_key: str) -> Dict[str, Dict[str, Any]]:
        preset = self.typography_presets.get(preset_key, self.typography_presets["warm_dessert"])
        base = self.typography_presets["warm_dessert"]
        result = {
            "title": {**base["title"], **preset.get("title", {})},
            "subtitle": {**base["subtitle"], **preset.get("subtitle", {})},
            "body": {**base["body"], **preset.get("body", {})},
            "cta": {**base["cta"], **preset.get("cta", {})},
            "badge": {**base["badge"], **preset.get("badge", {})},
        }
        return result

    def _build_badge_text(self, brief: Dict[str, Any], category: str) -> str:
        if brief.get("badge"):
            return brief["badge"]
        if category in ("dessert", "bakery"):
            return "Chef's Pick"
        if category in ("coffee", "tea"):
            return "Daily Roast"
        if category in ("beverage", "juice"):
            return "Fresh Blend"
        return "Limited Offer"

__all__ = ["StylePlanner", "StylePlan"]

