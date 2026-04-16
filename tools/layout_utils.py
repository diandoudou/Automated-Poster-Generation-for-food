from __future__ import annotations

import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

@dataclass(frozen=True)
class LayoutRegion:
    x: int
    y: int
    width: int
    height: int

@dataclass(frozen=True)
class TextRegion:
    role: str
    x: int
    y: int
    max_width: int
    font_size: int
    align: str = "left"
    line_spacing: int | None = None
    weight: str = "regular"
    uppercase: bool = False

@dataclass(frozen=True)
class LayoutTemplate:
    name: str
    description: str
    canvas_size: Tuple[int, int]
    image_region: LayoutRegion
    text_regions: Tuple[TextRegion, ...] = field(default_factory=tuple)
    background_mode: str = "solid"
    background_kwargs: Dict[str, Any] = field(default_factory=dict)
    decorations: Tuple[str, ...] = field(default_factory=tuple)

def _soft_glow_template() -> LayoutTemplate:
    return LayoutTemplate(
        name="soft_glow_portrait",
        description="Vertical layout / Creamy gradient + centered focus / High-end dessert",
        canvas_size=(1024, 1536),
        image_region=LayoutRegion(x=180, y=420, width=660, height=600),
        text_regions=(
            TextRegion("badge", x=100, y=100, max_width=300, font_size=32, uppercase=True),
            TextRegion("title", x=80, y=220, max_width=864, font_size=80, align="center", weight="bold"),
            TextRegion("subtitle", x=100, y=340, max_width=600, font_size=48),
            TextRegion("features", x=212, y=1080, max_width=600, font_size=38, line_spacing=52, align="center"),
            TextRegion("contact", x=750, y=1480, max_width=250, font_size=26, align="right"),
        ),
        background_mode="gradient",
        background_kwargs={"direction": "vertical"},
        decorations=("grain", "social_icons", "decorative_lines", "soft_bubbles"),
    )


def _dual_tone_template() -> LayoutTemplate:
    return LayoutTemplate(
        name="dual_tone_diagonal",
        description="Vertical layout / Diagonal two-color background + Modern beverages",
        canvas_size=(1024, 1536),
        image_region=LayoutRegion(x=180, y=420, width=660, height=600),
        text_regions=(
            TextRegion("badge", x=100, y=100, max_width=300, font_size=32, uppercase=True),
            TextRegion("title", x=80, y=220, max_width=864, font_size=80, align="center", weight="bold"),
            TextRegion("subtitle", x=100, y=340, max_width=600, font_size=48),
            TextRegion("features", x=212, y=1080, max_width=600, font_size=38, line_spacing=52, align="center"),
            TextRegion("contact", x=750, y=1480, max_width=250, font_size=26, align="right"),
        ),
        background_mode="dual_tone",
        background_kwargs={"angle": 60},
        decorations=("diagonal_stripes", "grain", "social_icons", "decorative_lines", "minimal_crosses"),
    )


def _spotlight_template() -> LayoutTemplate:
    return LayoutTemplate(
        name="spotlight_dark",
        description="Vertical format / Dark stage + Spotlight / High-end banquet",
        canvas_size=(1024, 1536),
        image_region=LayoutRegion(x=180, y=420, width=660, height=600),
        text_regions=(
            TextRegion("badge", x=100, y=100, max_width=300, font_size=32, uppercase=True),
            TextRegion("title", x=80, y=220, max_width=864, font_size=80, align="center", weight="bold"),
            TextRegion("subtitle", x=100, y=340, max_width=600, font_size=48),
            TextRegion("features", x=212, y=1080, max_width=600, font_size=38, line_spacing=52, align="center"),
            TextRegion("contact", x=750, y=1480, max_width=250, font_size=26, align="right"),
        ),
        background_mode="spotlight",
        background_kwargs={
            "spot_radius": 640,
            "vignette_strength": 0.72,
        },
        decorations=("grain", "social_icons", "decorative_lines", "radial_lines"),
    )


def _asym_focus_template() -> LayoutTemplate:
    return LayoutTemplate(
        name="asym_focus_left",
        description="Vertical format / Paper texture / Premium coffee",
        canvas_size=(1024, 1536),
        image_region=LayoutRegion(x=180, y=420, width=660, height=600),
        text_regions=(
            TextRegion("badge", x=100, y=100, max_width=300, font_size=32, uppercase=True),
            TextRegion("title", x=80, y=220, max_width=864, font_size=80, align="center", weight="bold"),
            TextRegion("subtitle", x=100, y=340, max_width=600, font_size=48),
            TextRegion("features", x=212, y=1080, max_width=600, font_size=38, line_spacing=52, align="center"),
            TextRegion("contact", x=750, y=1480, max_width=250, font_size=26, align="right"),
        ),
        background_mode="paper",
        background_kwargs={"grain_strength": 0.16},
        decorations=("grain", "social_icons", "decorative_lines", "coffee_beans"),
    )


def _collage_template() -> LayoutTemplate:
    return LayoutTemplate(
        name="collage_stack",
        description="Vertical format / Fabric texture / Refreshing beverage",
        canvas_size=(1024, 1536),
        image_region=LayoutRegion(x=180, y=420, width=660, height=600),
        text_regions=(
            TextRegion("badge", x=100, y=100, max_width=300, font_size=32, uppercase=True),
            TextRegion("title", x=80, y=220, max_width=864, font_size=80, align="center", weight="bold"),
            TextRegion("subtitle", x=100, y=340, max_width=600, font_size=48),
            TextRegion("features", x=212, y=1080, max_width=600, font_size=38, line_spacing=52, align="center"),
            TextRegion("contact", x=750, y=1480, max_width=250, font_size=26, align="right"),
        ),
        background_mode="fabric",
        background_kwargs={"direction": "diagonal"},
        decorations=("grain", "social_icons", "decorative_lines", "dot_grid"),
    )


def _grid_info_template() -> LayoutTemplate:
    return LayoutTemplate(
        name="grid_info_rich",
        description="Rich Information Type / Gradual Background / Promotional Poster",
        canvas_size=(1024, 1536),
        image_region=LayoutRegion(x=180, y=420, width=660, height=600),
        text_regions=(
            TextRegion("badge", x=100, y=100, max_width=300, font_size=32, uppercase=True),
            TextRegion("title", x=80, y=220, max_width=864, font_size=80, align="center", weight="bold"),
            TextRegion("subtitle", x=100, y=340, max_width=600, font_size=48),
            TextRegion("features", x=212, y=1080, max_width=600, font_size=38, line_spacing=52, align="center"),
            TextRegion("contact", x=750, y=1480, max_width=250, font_size=26, align="right"),
        ),
        background_mode="gradient",
        background_kwargs={"direction": "vertical"},
        decorations=("geometric_shapes", "grain", "social_icons", "decorative_lines", "info_badges"),
    )


def _diagonal_split_template() -> LayoutTemplate:
    return LayoutTemplate(
        name="diagonal_split",
        description="Diagonal division / Two-tone / Modern vitality",
        canvas_size=(1024, 1536),
        image_region=LayoutRegion(x=180, y=420, width=660, height=600),
        text_regions=(
            TextRegion("badge", x=100, y=100, max_width=300, font_size=32, uppercase=True),
            TextRegion("title", x=80, y=220, max_width=864, font_size=80, align="center", weight="bold"),
            TextRegion("subtitle", x=100, y=340, max_width=600, font_size=48),
            TextRegion("features", x=212, y=1080, max_width=600, font_size=38, line_spacing=52, align="center"),
            TextRegion("contact", x=750, y=1480, max_width=250, font_size=26, align="right"),
        ),
        background_mode="dual_tone",
        background_kwargs={"angle": 55},
        decorations=("grain", "social_icons", "decorative_lines", "minimal_crosses"),
    )


def _vibrant_collage_template() -> LayoutTemplate:
    return LayoutTemplate(
        name="vibrant_collage",
        description="Vibrant collage / Textured paper / Youthful",
        canvas_size=(1024, 1536),
        image_region=LayoutRegion(x=180, y=420, width=660, height=600),
        text_regions=(
            TextRegion("badge", x=100, y=100, max_width=300, font_size=32, uppercase=True),
            TextRegion("title", x=80, y=220, max_width=864, font_size=80, align="center", weight="bold"),
            TextRegion("subtitle", x=100, y=340, max_width=600, font_size=48),
            TextRegion("features", x=212, y=1080, max_width=600, font_size=38, line_spacing=52, align="center"),
            TextRegion("contact", x=750, y=1480, max_width=250, font_size=26, align="right"),
        ),
        background_mode="paper",
        background_kwargs={"grain_strength": 0.12},
        decorations=("geometric_shapes", "grain", "social_icons", "decorative_lines", "dot_grid"),
    )


BUILTIN_TEMPLATES: Dict[str, LayoutTemplate] = {
    "soft_glow_portrait": _soft_glow_template(),
    "dual_tone_diagonal": _dual_tone_template(),
    "spotlight_dark": _spotlight_template(),
    "asym_focus_left": _asym_focus_template(),
    "collage_stack": _collage_template(),
    "grid_info_rich": _grid_info_template(),
    "diagonal_split": _diagonal_split_template(),
    "vibrant_collage": _vibrant_collage_template(),
}

class PosterComposer:
    def __init__(self, font_overrides: Dict[str, str] | None = None):
        self.font_overrides = font_overrides or {}
        self._font_cache: Dict[Tuple[str, str, str, int], ImageFont.FreeTypeFont] = {}

        # Initialize cross-platform font library
        self._init_font_library()

    def _init_font_library(self):
        """Initialize font library based on operating system"""
        import platform
        system = platform.system()

        if system == "Darwin":  # macOS
            self.font_library: Dict[str, List[Path]] = {
                "serif_elegant": [
                    Path("assets/fonts/CormorantGaramond-Bold.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
                    Path("/System/Library/Fonts/Times.ttc"),
                    Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
                ],
                "sans_clean": [
                    Path("assets/fonts/NotoSans-Regular.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                    Path("/System/Library/Fonts/Helvetica.ttc"),
                    Path("/Library/Fonts/Arial.ttf"),
                ],
                "sans_bold": [
                    Path("assets/fonts/NotoSans-Bold.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
                    Path("/System/Library/Fonts/Helvetica.ttc"),
                    Path("/System/Library/Fonts/PingFang.ttc"),
                ],
                "script": [
                    Path("assets/fonts/GreatVibes-Regular.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Brush Script.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Snell Roundhand.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"),
                ],
                "script_bold": [
                    Path("/System/Library/Fonts/Supplemental/Brush Script.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Snell Roundhand.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Zapfino.ttf"),
                ],
                "decorative": [
                    Path("/System/Library/Fonts/Supplemental/Impact.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Copperplate.ttc"),
                    Path("/System/Library/Fonts/Supplemental/Party LET.ttf"),
                    Path("/System/Library/Fonts/Supplemental/Chalkboard.ttc"),
                ],
                "cjk_serif": [
                    Path("/System/Library/Fonts/PingFang.ttc"),
                    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
                    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
                ],
            }

            self._fallback_fonts = [
                Path(self.font_overrides.get("preferred") or ""),
                Path("/System/Library/Fonts/PingFang.ttc"),
                Path("/System/Library/Fonts/Helvetica.ttc"),
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            ]

        elif system == "Windows":
            self.font_library: Dict[str, List[Path]] = {
                "serif_elegant": [
                    Path("assets/fonts/CormorantGaramond-Bold.ttf"),
                    Path("C:/Windows/Fonts/cambria.ttc"),
                    Path("C:/Windows/Fonts/georgia.ttf"),
                    Path("C:/Windows/Fonts/times.ttf"),
                ],
                "sans_clean": [
                    Path("assets/fonts/NotoSans-Regular.ttf"),
                    Path("C:/Windows/Fonts/arial.ttf"),
                    Path("C:/Windows/Fonts/calibri.ttf"),
                    Path("C:/Windows/Fonts/segoeui.ttf"),
                ],
                "sans_bold": [
                    Path("assets/fonts/NotoSans-Bold.ttf"),
                    Path("C:/Windows/Fonts/arialbd.ttf"),
                    Path("C:/Windows/Fonts/calibrib.ttf"),
                    Path("C:/Windows/Fonts/msyhbd.ttc"),
                ],
                "script": [
                    Path("assets/fonts/GreatVibes-Regular.ttf"),
                    Path("C:/Windows/Fonts/segoesc.ttf"),
                    Path("C:/Windows/Fonts/BRUSHSCI.TTF"),
                    Path("C:/Windows/Fonts/FREESCPT.TTF"),
                ],
                "script_bold": [
                    Path("C:/Windows/Fonts/BRUSHSCI.TTF"),
                    Path("C:/Windows/Fonts/MISTRAL.TTF"),
                    Path("C:/Windows/Fonts/VLADIMIR.TTF"),
                    Path("C:/Windows/Fonts/KUNSTLER.TTF"),
                ],
                "decorative": [
                    Path("C:/Windows/Fonts/KUNSTLER.TTF"),
                    Path("C:/Windows/Fonts/impact.ttf"),
                    Path("C:/Windows/Fonts/ITCBLKAD.TTF"),
                    Path("C:/Windows/Fonts/LATINWD.TTF"),
                ],
                "cjk_serif": [
                    Path("C:/Windows/Fonts/msyh.ttc"),
                    Path("C:/Windows/Fonts/msyhbd.ttc"),
                    Path("C:/Windows/Fonts/simsun.ttc"),
                ],
            }

            self._fallback_fonts = [
                Path(self.font_overrides.get("preferred") or ""),
                Path("C:/Windows/Fonts/msyh.ttc"),
                Path("C:/Windows/Fonts/arial.ttf"),
            ]

        else:  # Linux
            self.font_library: Dict[str, List[Path]] = {
                "serif_elegant": [
                    Path("assets/fonts/CormorantGaramond-Bold.ttf"),
                    Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
                ],
                "sans_clean": [
                    Path("assets/fonts/NotoSans-Regular.ttf"),
                    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                ],
                "sans_bold": [
                    Path("assets/fonts/NotoSans-Bold.ttf"),
                    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                ],
                "script": [
                    Path("assets/fonts/GreatVibes-Regular.ttf"),
                    Path("/usr/share/fonts/truetype/uralic/URWChanceryL-MediItal.ttf"),
                ],
                "script_bold": [
                    Path("/usr/share/fonts/truetype/uralic/URWChanceryL-MediItal.ttf"),
                ],
                "decorative": [
                    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
                ],
                "cjk_serif": [
                    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
                ],
            }

            self._fallback_fonts = [
                Path(self.font_overrides.get("preferred") or ""),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            ]

        # Filter out empty paths from fallback
        self._fallback_fonts = [
            p for p in self._fallback_fonts
            if p and str(p) not in ("", ".")
        ]

        print(f"✅ PosterComposer font library initialized for {system}")

    def compose(
        self,
        subject_image_path: Path,
        template: LayoutTemplate,
        palette: Dict[str, str],
        text_blocks: List[Dict],
        output_path: Path,
        decor_context: Dict[str, Any] | None = None,
    ) -> Path:
        canvas = self._create_background(template, palette)
        self._apply_decorations(canvas, template, palette, decor_context)

        subject = Image.open(subject_image_path).convert("RGBA")
        fitted_subject, offset = self._fit_subject(subject, template.image_region)
        self._apply_subject_shadow(canvas, fitted_subject, offset)
        canvas.alpha_composite(fitted_subject, dest=offset)

        draw = ImageDraw.Draw(canvas)
        for block in text_blocks:
            self._draw_text_block(draw, block, palette)

        canvas = canvas.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="PNG")
        return output_path

    # Text rendering helpers
    def _draw_text_block(self, draw: ImageDraw.ImageDraw, block: Dict, palette: Dict[str, str]) -> None:
        content = block.get("text", "")
        items = block.get("items")
        if not content and not items:
            return

        font_size = block.get("font_size", 48)
        weight = block.get("weight", "regular")
        family = block.get("font_family", "sans_clean")
        role = block.get("role", "")
        font = self._get_font(family, weight, font_size, role)

        text_color = block.get("color") or palette.get("text_primary", "#1E1E1E")
        max_width = block.get("max_width", 600)
        align = block.get("align", "left")
        x = block.get("x", 80)
        y = block.get("y", 80)
        uppercase = block.get("uppercase", False)
        line_spacing = block.get("line_spacing") or int(font_size * 1.25)

        lines: List[str] = []
        if items:
            bullet = block.get("bullet", "• ")
            for item in items:
                text = item.upper() if uppercase else item
                lines.extend(self._wrap_text(f"{bullet}{text}", font, max_width))
        else:
            text = content.upper() if uppercase else content
            if block.get("force_single_line", False):
                text_width = self._measure_text(font, text, draw)
                original_font_size = font_size
                min_font_size = int(original_font_size * 0.5)

                while text_width > max_width and font_size > min_font_size:
                    font_size = max(font_size - 2, min_font_size)
                    font = self._get_font(family, weight, font_size, role)
                    text_width = self._measure_text(font, text, draw)

                block["font_size"] = font_size
                lines = [text]
            else:
                lines = self._wrap_text(text, font, max_width)

        if not lines:
            return

        actual_font_size = block.get("font_size", font_size)
        line_spacing = block.get("line_spacing") or int(actual_font_size * 1.25)

        max_line_width = max(self._measure_text(font, line, draw) for line in lines)
        block_height = len(lines) * line_spacing - (line_spacing - actual_font_size)
        block_width = min(max_width, max_line_width)
        start_x = self._align_x(x, block_width, max_width, align)

        if block.get("bg_color"):
            padding = block.get("padding", 18)
            radius = block.get("border_radius", 32)
            bg_rect = [
                start_x - padding,
                y - padding,
                start_x + block_width + padding,
                y + block_height + padding,
            ]
            bg_color = self._parse_color(block["bg_color"], block.get("bg_alpha", 220))
            draw.rounded_rectangle(bg_rect, radius=radius, fill=bg_color)
            if block.get("border_color"):
                border_color = self._parse_color(block["border_color"], block.get("border_alpha", 255))
                draw.rounded_rectangle(
                    bg_rect,
                    radius=radius,
                    outline=border_color,
                    width=block.get("border_width", 3),
                )

        y_cursor = y
        for line in lines:
            text_w = self._measure_text(font, line, draw)
            offset_x = self._align_x(x, text_w, max_width, align)
            if block.get("text_shadow"):
                shadow_color = self._parse_color(block.get("shadow_color", "#000000"), block.get("shadow_alpha", 80))
                draw.text(
                    (offset_x + block.get("shadow_offset_x", 3), y_cursor + block.get("shadow_offset_y", 3)),
                    line,
                    fill=shadow_color,
                    font=font,
                )
            draw.text((offset_x, y_cursor), line, fill=self._parse_color(text_color), font=font)
            y_cursor += line_spacing

    def _align_x(self, x: int, actual_width: int, max_width: int, align: str) -> int:
        if align == "center":
            return x + max(0, (max_width - actual_width) // 2)
        if align == "right":
            return x + max(0, max_width - actual_width)
        return x

    def _get_font(self, family: str, weight: str, size: int, role: str = "") -> ImageFont.FreeTypeFont:
        """Get font with cross-platform support"""
        # Map roles to font families
        if role == "title":
            family = "script"
        elif role == "badge":
            family = "sans_bold"
        elif role == "features":
            family = "script"

        # Check cache
        key = (family, weight, size, role)
        if key in self._font_cache:
            return self._font_cache[key]

        candidate_paths: List[Path] = []

        # Check user overrides first
        override_key = f"{family}_{weight}"
        if override_key in self.font_overrides:
            override_path = self.font_overrides[override_key]
            if override_path:  # Only add if not empty
                candidate_paths.append(Path(override_path))

        # Add font library paths
        if family in self.font_library:
            candidate_paths.extend(self.font_library[family])

        # Add fallback fonts
        candidate_paths.extend(self._fallback_fonts)

        # Filter out invalid paths
        candidate_paths = [
            p for p in candidate_paths
            if p and str(p) not in ("", ".") and not (p.exists() and p.is_dir())
        ]

        # Try to load each candidate
        for path in candidate_paths:
            if not path.exists():
                continue

            try:
                font = ImageFont.truetype(str(path), size=size)
                self._font_cache[key] = font
                return font
            except OSError as e:
                # Only warn for actual font files
                if path.suffix.lower() in ['.ttf', '.ttc', '.otf']:
                    print(f"⚠️  Failed to load {path.name}: {e}")
                continue

        # Final fallback
        print(f"⚠️  No font found for {family}/{weight}, using default")
        font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        if not text:
            return []

        paragraphs = text.split('\n')
        all_lines: List[str] = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                all_lines.append("")
                continue

            tokens = paragraph.split()
            if len(tokens) == 1:
                tokens = list(paragraph)

            current = ""
            for token in tokens:
                candidate = f"{current} {token}".strip() if current else token
                width = self._measure_text(font, candidate)
                if width <= max_width:
                    current = candidate
                else:
                    if current:
                        all_lines.append(current)
                    current = token
            if current:
                all_lines.append(current)

        return all_lines

    @staticmethod
    def _measure_text(font: ImageFont.FreeTypeFont, text: str, draw: ImageDraw.ImageDraw | None = None) -> int:
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        if draw is not None and hasattr(draw, "textlength"):
            return int(draw.textlength(text, font=font))
        return font.getsize(text)[0]

    # Subject & background helpers
    def _fit_subject(self, subject: Image.Image, region: LayoutRegion) -> Tuple[Image.Image, Tuple[int, int]]:
        target_w, target_h = region.width, region.height
        subject_ratio = subject.width / subject.height
        target_ratio = target_w / target_h

        if subject_ratio > target_ratio:
            new_w = target_w
            new_h = int(target_w / subject_ratio)
        else:
            new_h = target_h
            new_w = int(target_h * subject_ratio)

        resized = subject.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)
        paste_x = region.x + (region.width - resized.width) // 2
        paste_y = region.y + (region.height - resized.height) // 2
        return resized, (paste_x, paste_y)

    def _apply_subject_shadow(self, canvas: Image.Image, subject: Image.Image, offset: Tuple[int, int]) -> None:
        if subject.mode != "RGBA":
            return

        alpha = subject.split()[-1]

        shadow_far = alpha.filter(ImageFilter.GaussianBlur(40))
        shadow_far_img = Image.new("RGBA", subject.size, (20, 10, 5, 50))  # 深棕色
        shadow_far_img.putalpha(shadow_far)
        shift_far = (offset[0] + 30, offset[1] + 45)
        canvas.alpha_composite(shadow_far_img, dest=shift_far)

        shadow_mid = alpha.filter(ImageFilter.GaussianBlur(20))
        shadow_mid_img = Image.new("RGBA", subject.size, (15, 10, 8, 70))  # 中等棕黑色
        shadow_mid_img.putalpha(shadow_mid)
        shift_mid = (offset[0] + 18, offset[1] + 28)
        canvas.alpha_composite(shadow_mid_img, dest=shift_mid)

        shadow_near = alpha.filter(ImageFilter.GaussianBlur(8))
        shadow_near_img = Image.new("RGBA", subject.size, (25, 20, 15, 90))  # 浅棕色
        shadow_near_img.putalpha(shadow_near)
        shift_near = (offset[0] + 8, offset[1] + 12)
        canvas.alpha_composite(shadow_near_img, dest=shift_near)

    def _create_background(self, template: LayoutTemplate, palette: Dict[str, str]) -> Image.Image:
        mode = template.background_mode
        kwargs = template.background_kwargs or {}
        size = template.canvas_size
        primary = palette.get("background", "#F5F5F0")
        secondary = palette.get("background_alt", primary)

        if mode is None:
            raise ValueError("Background mode cannot be None.")

        if mode == "gradient":
            canvas = self._linear_gradient(size, primary, secondary, kwargs.get("direction", "vertical"))
        elif mode == "dual_tone":
            canvas = self._dual_tone_background(size, primary, secondary, kwargs.get("angle", 45))
        elif mode == "spotlight":
            canvas = self._spotlight_background(
                size,
                primary,
                secondary,
                kwargs.get("spot_radius", min(size) * 0.7),
                kwargs.get("vignette_strength", 0.5),
            )
        elif mode == "paper":
            canvas = self._paper_texture_background(size, primary, kwargs.get("grain_strength", 0.1))
        elif mode == "fabric":
            canvas = self._fabric_background(size, primary, secondary, kwargs.get("direction", "horizontal"))
        else:
            canvas = Image.new("RGBA", size, primary)
        return canvas.convert("RGBA")

    def _apply_decorations(
        self,
        canvas: Image.Image,
        template: LayoutTemplate,
        palette: Dict[str, str],
        decor_context: Dict[str, Any] | None = None,
    ) -> None:
        decorators = {
            "radial_glow": self._decor_radial_glow,
            "grain": self._decor_grain,
            "accent_rings": self._decor_accent_rings,
            "diagonal_stripes": self._decor_diagonal_stripes,
            "sparkles": self._decor_sparkles,
            "table_reflection": self._decor_table_reflection,
            "flow_lines": self._decor_flow_lines,
            "food_motifs": self._decor_food_motifs,
            "foreground_bokeh": self._decor_foreground_bokeh,
            "stack_cards": self._decor_stack_cards,
            "geometric_shapes": self._decor_geometric_shapes,
            "social_icons": self._decor_social_icons,
            "qr_code": self._decor_qr_code,
            "decorative_lines": self._decor_decorative_lines,
            "corner_ornaments": self._decor_corner_ornaments,
            "checkered_pattern": self._decor_checkered_pattern,
            "ribbon_banner": self._decor_ribbon_banner,
            "food_icons": self._decor_food_icons,
        }

        ctx = decor_context or {}
        for name in template.decorations:
            func = decorators.get(name)
            if func:
                func(canvas, template, palette, ctx)

        for extra in ctx.get("extra_decor", []):
            func = decorators.get(extra)
            if func:
                func(canvas, template, palette, ctx)

    def _linear_gradient(self, size, color_a, color_b, direction):
        base = Image.new("RGBA", size, color_a)
        top = Image.new("RGBA", size, color_b)
        gradient = Image.linear_gradient("L")
        if direction == "horizontal":
            gradient = gradient.rotate(90, expand=True)
        elif direction == "diagonal":
            gradient = gradient.rotate(45, expand=True)
        mask = gradient.resize(size)
        return Image.composite(top, base, mask)

    def _dual_tone_background(self, size, color_a, color_b, angle):
        canvas = Image.new("RGBA", size, color_a)
        overlay = Image.new("RGBA", size, color_b)
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        radians = math.radians(angle)
        offset = math.tan(radians) * size[1]
        points = [
            (-offset, size[1]),
            (size[0], 0),
            (size[0] + offset, 0),
            (0, size[1]),
        ]
        draw.polygon(points, fill=255)
        return Image.composite(overlay, canvas, mask)

    def _spotlight_background(self, size, color_dark, color_light, radius, vignette_strength):
        base = Image.new("RGBA", size, color_dark)
        center = (size[0] // 2, int(size[1] * 0.4))
        glow = Image.new("L", size, 0)
        draw = ImageDraw.Draw(glow)
        draw.ellipse(
            [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
            fill=255,
        )
        glow = glow.filter(ImageFilter.GaussianBlur(radius * 0.6))
        glow_colored = Image.new("RGBA", size, color_light)
        spotlight = Image.composite(glow_colored, base, glow)

        vignette = Image.new("L", size, int(255 * vignette_strength))
        draw_v = ImageDraw.Draw(vignette)
        draw_v.rectangle([0, 0, size[0], size[1]], fill=int(255 * vignette_strength))
        draw_v.ellipse([size[0] * 0.1, size[1] * 0.2, size[0] * 0.9, size[1] * 0.95], fill=0)
        vignette = vignette.filter(ImageFilter.GaussianBlur(120))
        dark_layer = Image.new("RGBA", size, (0, 0, 0))
        spotlight = Image.composite(dark_layer, spotlight, vignette)
        return spotlight

    def _paper_texture_background(self, size, base_color, grain_strength):
        base = Image.new("RGBA", size, base_color)
        noise = np.random.normal(128, 40, (size[1], size[0])).clip(0, 255).astype(np.uint8)
        noise_img = Image.fromarray(noise, mode="L").filter(ImageFilter.GaussianBlur(1.2))
        overlay = Image.merge("RGBA", (noise_img, noise_img, noise_img, noise_img.point(lambda _: int(255 * grain_strength))))
        base.alpha_composite(overlay)
        fibers = Image.new("L", size, 0)
        draw = ImageDraw.Draw(fibers)
        for y in range(0, size[1], 8):
            intensity = random.randint(10, 25)
            draw.line([(0, y), (size[0], y)], fill=intensity, width=1)
        fibers = fibers.filter(ImageFilter.GaussianBlur(0.8))
        fiber_overlay = Image.new("RGBA", size, base_color)
        fiber_overlay.putalpha(fibers)
        base = Image.composite(fiber_overlay, base, fibers)
        return base

    def _fabric_background(self, size, color_a, color_b, direction):
        base = self._linear_gradient(size, color_a, color_b, "diagonal" if direction == "diagonal" else "horizontal")
        weave = Image.new("L", size, 0)
        draw = ImageDraw.Draw(weave)
        spacing = 12
        for x in range(0, size[0], spacing):
            draw.line([(x, 0), (x, size[1])], fill=80, width=2)
        for y in range(0, size[1], spacing):
            draw.line([(0, y), (size[0], y)], fill=50, width=2)
        weave = weave.filter(ImageFilter.GaussianBlur(2))
        overlay = Image.new("RGBA", size, "#FFFFFF")
        overlay.putalpha(weave.point(lambda v: int(v * 0.4)))
        base.alpha_composite(overlay)
        return base

    def _decor_radial_glow(self, canvas, template, palette, ctx):
        w, h = canvas.size
        radius = template.background_kwargs.get("glow_radius", min(w, h) * 0.7)
        center = (
            template.background_kwargs.get("glow_center_x", w // 2),
            template.background_kwargs.get("glow_center_y", int(h * 0.35)),
        )
        glow = Image.new("L", canvas.size, 0)
        draw = ImageDraw.Draw(glow)
        draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=255)
        glow = glow.filter(ImageFilter.GaussianBlur(radius * 0.4))
        color = palette.get("accent", "#FFFFFF")
        overlay = Image.new("RGBA", canvas.size, color)
        overlay.putalpha(glow)
        canvas.alpha_composite(overlay)

    def _decor_grain(self, canvas, template, palette, ctx):
        w, h = canvas.size
        noise = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        noise_img = Image.fromarray(noise, mode="L").filter(ImageFilter.GaussianBlur(0.8))
        alpha = int(255 * 0.08)
        noise_rgba = Image.merge("RGBA", (noise_img, noise_img, noise_img, noise_img.point(lambda _: alpha)))
        canvas.alpha_composite(noise_rgba)

    def _decor_accent_rings(self, canvas, template, palette, ctx):
        draw = ImageDraw.Draw(canvas)
        region = template.image_region
        center = (region.x + region.width // 2, region.y + region.height // 2)
        max_radius = int(region.width * 0.65)
        color = palette.get("accent", "#FFD9A0")
        for scale in (1.1, 1.3, 1.5):
            radius = int(max_radius * scale)
            bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
            draw.arc(bbox, start=200, end=340, fill=color, width=6)

    def _decor_diagonal_stripes(self, canvas, template, palette, ctx):
        stripe_color = palette.get("accent_alt", "#FFFFFF")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        step = 140
        width = 16
        rgba = self._hex_with_alpha(stripe_color, 60)
        for offset in range(-canvas.height, canvas.width + canvas.height, step):
            draw.line([(offset, canvas.size[1]), (offset + canvas.size[1], 0)], fill=rgba, width=width)
        overlay = overlay.filter(ImageFilter.GaussianBlur(4))
        canvas.alpha_composite(overlay)

    def _decor_sparkles(self, canvas, template, palette, ctx):
        gold_color = "#F4D03F"

        draw = ImageDraw.Draw(canvas)

        star_positions = [
            (canvas.size[0] - 120, 150),
            (120, canvas.size[1] - 200),
        ]

        for x, y in star_positions:
            size = 25
            self._draw_star(draw, x, y, size, gold_color, opacity=180)

        num_small_stars = 25
        for _ in range(num_small_stars):
            x = random.randint(100, canvas.size[0] - 100)
            y = random.randint(150, canvas.size[1] - 150)
            size = random.randint(8, 15)
            opacity = random.randint(100, 160)
            self._draw_star(draw, x, y, size, gold_color, opacity)

    def _decor_table_reflection(self, canvas, template, palette, ctx):
        gradient = Image.new("L", (1, template.canvas_size[1]), color=0)
        for y in range(template.canvas_size[1]):
            value = int(max(0, 255 - (y - template.canvas_size[1] * 0.55)))
            gradient.putpixel((0, y), max(0, min(255, value)))
        gradient = gradient.resize(template.canvas_size)
        color = self._hex_with_alpha(palette.get("background_alt", "#FFFFFF"), 80)
        overlay = Image.new("RGBA", template.canvas_size, color)
        overlay.putalpha(gradient)
        canvas.alpha_composite(overlay)

    def _decor_flow_lines(self, canvas, template, palette, ctx):
        gold_color = "#F4D03F"  # 亮金色

        draw = ImageDraw.Draw(canvas)
        num_lines = random.randint(3, 4)
        for i in range(num_lines):
            points = []
            start_y = random.randint(int(canvas.size[1] * 0.2), int(canvas.size[1] * 0.8))
            amplitude = random.randint(15, 35)
            frequency = random.uniform(0.008, 0.012)
            phase = random.uniform(0, math.pi * 2)

            for x in range(0, canvas.size[0], 10):
                y = start_y + math.sin(x * frequency + phase) * amplitude
                points.append((x, int(y)))

            opacity = random.randint(100, 140)
            line_color = self._hex_with_alpha(gold_color, opacity)
            draw.line(points, fill=line_color, width=2)

    def _decor_food_motifs(self, canvas, template, palette, ctx):
        category = ctx.get("food_category", "generic")
        colors = [palette.get("accent"), palette.get("accent_alt"), palette.get("background_alt"), "#FFFFFF"]
        colors = [c for c in colors if c]
        draw = ImageDraw.Draw(canvas)
        count = ctx.get("motif_density", 6)
        for _ in range(count):
            color = random.choice(colors)
            if category in ("dessert", "bakery"):
                self._draw_sprinkle(draw, template.canvas_size, color)
            elif category in ("coffee", "tea"):
                self._draw_leaf(draw, template.canvas_size, color)
            elif category in ("beverage", "juice"):
                self._draw_citrus_slice(draw, template.canvas_size, color)
            else:
                self._draw_generic_curve(draw, template.canvas_size, color)

    def _decor_foreground_bokeh(self, canvas, template, palette, ctx):
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        base_color = palette.get("background_alt", "#FFFFFF")
        for _ in range(25):
            radius = random.randint(40, 120)
            x = random.randint(-radius, canvas.size[0] + radius)
            y = random.randint(int(canvas.size[1] * 0.2), canvas.size[1] + radius)
            alpha = random.randint(30, 70)
            color = self._hex_with_alpha(base_color, alpha)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        overlay = overlay.filter(ImageFilter.GaussianBlur(12))
        canvas.alpha_composite(overlay)

    def _decor_stack_cards(self, canvas, template, palette, ctx):
        draw = ImageDraw.Draw(canvas)
        base_region = template.image_region
        colors = [palette.get("accent_alt", "#FFFFFF"), palette.get("background_alt", "#CCCCCC")]
        offsets = [(-40, -40), (40, 30)]
        for (dx, dy), color in zip(offsets, colors):
            rect = [
                base_region.x + dx,
                base_region.y + dy,
                base_region.x + base_region.width + dx,
                base_region.y + base_region.height + dy,
            ]
            draw.rounded_rectangle(rect, radius=60, fill=self._hex_with_alpha(color, 120))

    def _draw_sprinkle(self, draw, size, color):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        length = random.randint(40, 90)
        angle = random.uniform(0, math.pi)
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length
        draw.line([(x, y), (x + dx, y + dy)], fill=self._hex_with_alpha(color, 160), width=8)

    def _draw_leaf(self, draw, size, color):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        length = random.randint(80, 140)
        width = random.randint(30, 60)
        bbox = [x - width, y - length, x + width, y + length]
        draw.ellipse(bbox, outline=self._hex_with_alpha(color, 120), width=6)
        draw.line([(x, y - length), (x, y + length)], fill=self._hex_with_alpha(color, 90), width=3)

    def _draw_citrus_slice(self, draw, size, color):
        radius = random.randint(40, 80)
        x = random.randint(radius, size[0] - radius)
        y = random.randint(radius, size[1] - radius)
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=self._hex_with_alpha(color, 160), width=5)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            dx = math.cos(rad) * radius
            dy = math.sin(rad) * radius
            draw.line([(x, y), (x + dx, y + dy)], fill=self._hex_with_alpha(color, 120), width=3)

    def _draw_generic_curve(self, draw, size, color):
        points = []
        start_y = random.randint(0, size[1])
        freq = random.uniform(0.001, 0.003)
        amp = random.randint(50, 100)
        for x in range(0, size[0], 25):
            y = start_y + math.sin(x * freq) * amp
            points.append((x, y))
        draw.line(points, fill=self._hex_with_alpha(color, 100), width=5)

    def _decor_geometric_shapes(self, canvas, template, palette, ctx):
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        colors = [
            palette.get("accent", "#FFD700"),
            palette.get("accent_alt", "#FFA500"),
            palette.get("background_alt", "#FFFFFF"),
        ]

        count = ctx.get("shape_count", 15)
        for _ in range(count):
            shape = random.choice(["circle", "triangle", "square"])
            color = self._hex_with_alpha(random.choice(colors), random.randint(40, 90))
            size = random.randint(25, 80)
            x = random.randint(size, canvas.size[0] - size)
            y = random.randint(size, canvas.size[1] - size)

            if shape == "circle":
                draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
            elif shape == "triangle":
                points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
                draw.polygon(points, fill=color)
            elif shape == "square":
                points = [(x, y-size), (x+size, y), (x, y+size), (x-size, y)]
                draw.polygon(points, fill=color)

        overlay = overlay.filter(ImageFilter.GaussianBlur(3))
        canvas.alpha_composite(overlay)

    def _decor_qr_code(self, canvas, template, palette, ctx):
        try:
            import qrcode
            qr = qrcode.QRCode(box_size=6, border=1)
            qr.add_data(ctx.get("qr_content", "https://example.com/menu"))
            qr.make()
            qr_img = qr.make_image(
                fill_color=palette.get("text_primary", "#000000"),
                back_color="white"
            ).convert("RGBA")
            qr_img = qr_img.resize((160, 160), Image.LANCZOS)

            qr_bg = Image.new("RGBA", (180, 180), "white")
            qr_bg.paste(qr_img, (10, 10))

            position = (70, canvas.size[1] - 250)
            canvas.paste(qr_bg, position)

            draw = ImageDraw.Draw(canvas)
            font = self._get_font("sans_clean", "regular", 20)
            text = "SCAN HERE"
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = position[0] + (180 - text_width) // 2
            draw.text((text_x, position[1] + 190), text,
                     fill=self._parse_color(palette.get("text_primary", "#000000")),
                     font=font)
        except ImportError:
            draw = ImageDraw.Draw(canvas)
            rect = [70, canvas.size[1]-250, 250, canvas.size[1]-70]
            draw.rectangle(rect, fill="#FFFFFF", outline=palette.get("text_primary"), width=3)
            font = self._get_font("sans_clean", "regular", 16)
            draw.text((110, canvas.size[1]-170), "QR CODE",
                     fill=self._parse_color(palette.get("text_primary")), font=font)

    def _decor_social_icons(self, canvas, template, palette, ctx):
        draw = ImageDraw.Draw(canvas)
        icon_size = 36
        spacing = 55
        platforms = ["f", "t", "@", "w"]

        total_width = len(platforms) * spacing - (spacing - icon_size)
        start_x = (canvas.size[0] - total_width) // 2
        y = canvas.size[1] - 80

        icon_color = palette.get("accent", "#FFD700")
        text_color = palette.get("background", "#FFFFFF")

        for i, icon_char in enumerate(platforms):
            x = start_x + i * spacing
            draw.ellipse([x, y, x+icon_size, y+icon_size],
                        fill=self._hex_with_alpha(icon_color, 200),
                        outline=self._hex_with_alpha(palette.get("text_primary"), 150),
                        width=2)
            font = self._get_font("sans_bold", "bold", 20)
            text_bbox = font.getbbox(icon_char)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text((x + (icon_size - text_width)//2, y + (icon_size - text_height)//2 - 2),
                     icon_char, fill=self._parse_color(text_color), font=font)

    def _decor_decorative_lines(self, canvas, template, palette, ctx):
        draw = ImageDraw.Draw(canvas)
        color = palette.get("accent", "#FFD700")

        y_top = 55
        draw.line([(50, y_top), (canvas.size[0]-50, y_top)], fill=color, width=4)
        draw.line([(50, y_top+8), (canvas.size[0]-50, y_top+8)],
                 fill=self._hex_with_alpha(color, 80), width=2)

        y_bottom = canvas.size[1] - 140
        draw.line([(50, y_bottom), (canvas.size[0]-50, y_bottom)], fill=color, width=4)

    def _decor_corner_ornaments(self, canvas, template, palette, ctx):
        draw = ImageDraw.Draw(canvas)
        color = palette.get("accent", "#FFD700")
        size = 35

        corners = [
            (60, 60),
            (canvas.size[0]-60, 60),
            (60, canvas.size[1]-60),
            (canvas.size[0]-60, canvas.size[1]-60),
        ]

        for x, y in corners:
            draw.line([(x-size, y), (x+size, y)], fill=color, width=5)
            draw.line([(x, y-size), (x, y+size)], fill=color, width=5)
            dot_size = 6
            draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size],
                        fill=color)

    def _decor_checkered_pattern(self, canvas, template, palette, ctx):
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        square_size = 50
        color1 = self._hex_with_alpha(palette.get("accent", "#FFD700"), 30)
        color2 = self._hex_with_alpha(palette.get("background_alt", "#FFF"), 15)

        start_y = int(canvas.size[1] * 0.3)
        end_y = int(canvas.size[1] * 0.75)

        for y in range(start_y, end_y, square_size):
            for x in range(0, canvas.size[0], square_size):
                color = color1 if (x//square_size + y//square_size) % 2 == 0 else color2
                draw.rectangle([x, y, x+square_size, y+square_size], fill=color)

        overlay = overlay.filter(ImageFilter.GaussianBlur(2))
        canvas.alpha_composite(overlay)

    def _decor_ribbon_banner(self, canvas, template, palette, ctx):
        ribbon_color = palette.get("accent", "#FFD700")
        ribbon_height = 70
        y = ctx.get("ribbon_y", 160)

        ribbon_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ribbon_layer)

        draw.rectangle([0, y, canvas.size[0], y + ribbon_height],
                      fill=self._hex_with_alpha(ribbon_color, 180))

        draw.line([(0, y), (canvas.size[0], y)], fill="#FFFFFF", width=3)
        draw.line([(0, y + ribbon_height), (canvas.size[0], y + ribbon_height)],
                 fill="#FFFFFF", width=3)

        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rectangle([0, y + ribbon_height, canvas.size[0], y + ribbon_height + 10],
                              fill=self._hex_with_alpha("#000000", 50))
        shadow = shadow.filter(ImageFilter.GaussianBlur(6))

        canvas.alpha_composite(shadow)
        canvas.alpha_composite(ribbon_layer)

    def _decor_food_icons(self, canvas, template, palette, ctx):
        draw = ImageDraw.Draw(canvas)
        category = ctx.get("food_category", "generic")
        colors = [palette.get("accent"), palette.get("accent_alt")]
        colors = [c for c in colors if c]

        count = ctx.get("icon_count", 6)
        for _ in range(count):
            color = self._hex_with_alpha(random.choice(colors), random.randint(60, 100))
            x = random.randint(60, canvas.size[0] - 60)
            y = random.randint(60, canvas.size[1] - 60)

            if category in ("dessert", "bakery"):
                self._draw_star(draw, x, y, 30, color)
            elif category in ("coffee", "tea"):
                draw.ellipse([x-15, y-20, x+15, y+20], fill=color)
                draw.line([(x-5, y-10), (x+5, y+10)], fill=color, width=3)
            else:
                draw.ellipse([x-20, y-20, x+20, y+20],
                            outline=color, width=4)

    def _decor_soft_bubbles(self, canvas, template, palette, ctx):
        accent = palette.get("accent", "#FFE5CC")
        draw = ImageDraw.Draw(canvas)

        for _ in range(20):
            x = random.randint(50, canvas.size[0] - 50)
            y = random.randint(100, canvas.size[1] - 100)
            radius = random.randint(15, 45)
            opacity = random.randint(20, 50)
            color = self._hex_with_alpha(accent, opacity)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    def _decor_radial_lines(self, canvas, template, palette, ctx):
        center_x = canvas.size[0] // 2
        center_y = int(canvas.size[1] * 0.4)
        accent = palette.get("accent", "#FFD700")
        draw = ImageDraw.Draw(canvas)

        for i in range(12):
            angle = (i * 30) * math.pi / 180
            length = random.randint(200, 400)
            end_x = center_x + int(math.cos(angle) * length)
            end_y = center_y + int(math.sin(angle) * length)
            opacity = random.randint(15, 35)
            color = self._hex_with_alpha(accent, opacity)
            draw.line([(center_x, center_y), (end_x, end_y)], fill=color, width=1)

    def _decor_coffee_beans(self, canvas, template, palette, ctx):
        accent = palette.get("accent", "#8B6F47")
        draw = ImageDraw.Draw(canvas)

        for _ in range(15):
            x = random.randint(80, canvas.size[0] - 80)
            y = random.randint(120, canvas.size[1] - 120)
            size = random.randint(8, 16)
            opacity = random.randint(40, 80)
            color = self._hex_with_alpha(accent, opacity)

            draw.ellipse([x - size, y - size//2, x + size, y + size//2], fill=color)
            draw.line([(x - size//2, y), (x + size//2, y)], fill=color, width=2)

    def _decor_dot_grid(self, canvas, template, palette, ctx):
        accent = palette.get("accent", "#FF6B9D")
        draw = ImageDraw.Draw(canvas)

        spacing = 80
        for x in range(100, canvas.size[0] - 100, spacing):
            for y in range(150, canvas.size[1] - 150, spacing):
                if random.random() > 0.5:
                    radius = random.randint(3, 8)
                    opacity = random.randint(60, 120)
                    color = self._hex_with_alpha(accent, opacity)
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    def _decor_minimal_crosses(self, canvas, template, palette, ctx):
        accent = palette.get("accent", "#4A90E2")
        draw = ImageDraw.Draw(canvas)

        for _ in range(12):
            x = random.randint(100, canvas.size[0] - 100)
            y = random.randint(150, canvas.size[1] - 150)
            size = random.randint(8, 15)
            opacity = random.randint(80, 140)
            color = self._hex_with_alpha(accent, opacity)

            draw.line([(x - size, y), (x + size, y)], fill=color, width=2)
            draw.line([(x, y - size), (x, y + size)], fill=color, width=2)

    def _decor_info_badges(self, canvas, template, palette, ctx):
        accent = palette.get("accent", "#FF9500")
        draw = ImageDraw.Draw(canvas)

        positions = [
            (120, 180), (canvas.size[0] - 150, 200),
            (100, canvas.size[1] - 250), (canvas.size[0] - 120, canvas.size[1] - 300)
        ]

        for x, y in positions:
            opacity = random.randint(50, 90)
            color = self._hex_with_alpha(accent, opacity)
            width, height = random.randint(30, 50), random.randint(15, 25)
            draw.rounded_rectangle([x, y, x + width, y + height], radius=5, fill=color)

    def _draw_star(self, draw, cx, cy, size, color, opacity=200):
        outer_points = []
        for i in range(5):
            angle = math.radians(i * 144 - 90)
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            outer_points.append((x, y))

        fill_color = self._hex_with_alpha(color, opacity)
        draw.polygon(outer_points, fill=fill_color)

        inner_size = size * 0.4
        highlight_points = []
        for i in range(5):
            angle = math.radians(i * 144 - 90)
            x = cx + inner_size * math.cos(angle)
            y = cy + inner_size * math.sin(angle)
            highlight_points.append((x, y))

        highlight_color = self._hex_with_alpha("#FFFFFF", min(255, opacity + 60))
        draw.polygon(highlight_points, fill=highlight_color)

    def _parse_color(self, color: Any, alpha: int | None = None) -> Tuple[int, int, int, int]:
        if isinstance(color, tuple):
            if len(color) == 4:
                return color
            if len(color) == 3:
                return (*color, alpha or 255)
        if isinstance(color, str):
            color = color.lstrip("#")
            if len(color) == 6:
                r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
                return (r, g, b, alpha or 255)
        return (255, 255, 255, alpha or 255)

    def _hex_with_alpha(self, hex_color: str, alpha: int) -> Tuple[int, int, int, int]:
        return self._parse_color(hex_color, alpha)


__all__ = [
    "LayoutRegion",
    "TextRegion",
    "LayoutTemplate",
    "PosterComposer",
    "BUILTIN_TEMPLATES",
]

