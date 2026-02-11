"""
Thumbnail Generator - Professional YouTube thumbnails for LDS Ven Sígueme content.

Supports 7 layout presets for visual variety:
  luminoso, dramatico, celestial, profeta, esperanza, impacto, minimalista

Design principles (research-backed for 45-65+ Spanish-speaking LDS audience):
- Full-bleed sacred art image with configurable gradient overlays
- Large text with UPPERCASE words highlighted in accent color
- Scripture badge + "VEN SÍGUEME" brand badge
- High contrast (7:1+) for older demographic readability
- 3-5 words max in title for CTR optimization

Fonts recommended (download from Google Fonts → data/fonts/):
  - Montserrat-ExtraBold.ttf (primary headline)
  - Montserrat-Bold.ttf (badges, secondary text)
  - Cinzel-Bold.ttf (elegant serif alternative)
"""

import json
import math
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
FONTS_DIR = DATA_DIR / "fonts"
OUTPUT_DIR = DATA_DIR / "output"
SCRIPTS_DIR = DATA_DIR / "scripts"
CONFIG_PATH = DATA_DIR / "config.json"
THUMBNAIL_CONFIG_PATH = DATA_DIR / "thumbnail_config.json"

# Thumbnail dimensions (YouTube standard)
THUMB_WIDTH = 1280
THUMB_HEIGHT = 720


def hex_to_rgb(hex_color: str) -> Tuple[int, ...]:
    """Convert hex color to RGB or RGBA tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 8:
        # RGBA
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def load_config() -> dict:
    """Load main video configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_thumbnail_config() -> dict:
    """Load thumbnail-specific configuration."""
    with open(THUMBNAIL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_script() -> dict:
    """Load the current script with BOM handling."""
    script_path = SCRIPTS_DIR / "current.json"
    with open(script_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No valid JSON object found in script file")
    return json.loads(content[start_idx:end_idx + 1])


def get_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """
    Load a font by name with comprehensive fallback chain.

    Priority: data/fonts/{name} → Windows system fonts → PIL default
    """
    # Try project fonts directory
    font_path = FONTS_DIR / font_name
    if font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception:
            pass

    # Fallback chain for common fonts
    fallbacks = [
        FONTS_DIR / "Montserrat-ExtraBold.ttf",
        FONTS_DIR / "Montserrat-Bold.ttf",
        FONTS_DIR / "Oswald-SemiBold.ttf",
        FONTS_DIR / "EBGaramond-Bold.ttf",
        FONTS_DIR / "PlayfairDisplay-Bold.ttf",
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/georgiab.ttf"),
        Path("C:/Windows/Fonts/impact.ttf"),
    ]

    for fb in fallbacks:
        if fb.exists():
            try:
                return ImageFont.truetype(str(fb), size)
            except Exception:
                continue

    print(f"WARNING: Font '{font_name}' not found, using default")
    return ImageFont.load_default()


def find_best_image(script: dict, thumb_config: dict) -> Optional[Path]:
    """
    Select the best image for the thumbnail.

    Strategy:
    - If asset_id is specified in config, use that image
    - If custom_path is specified, use that
    - If 'auto', pick the first image (usually the most dramatic/opening scene)
    """
    image_cfg = thumb_config.get("image", {})
    source = image_cfg.get("source", "auto")

    # Custom path
    custom_path = image_cfg.get("custom_path")
    if custom_path and source == "custom":
        p = Path(custom_path)
        if p.exists():
            return p

    # Specific asset_id
    asset_id = image_cfg.get("asset_id")
    if asset_id and source == "asset":
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            p = IMAGES_DIR / f"{asset_id}{ext}"
            if p.exists():
                return p

    # Auto: pick the first available image from the script's visual assets
    script_data = script.get("script", {})
    narration = script_data.get("narration", {})
    visual_assets = narration.get("visual_assets", [])

    for asset in visual_assets:
        aid = asset.get("visual_asset_id")
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            p = IMAGES_DIR / f"{aid}{ext}"
            if p.exists():
                return p

    # Last resort: any image in the directory
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        images = list(IMAGES_DIR.glob(ext))
        if images:
            return images[0]

    return None


def crop_to_thumbnail(
    image: Image.Image,
    zoom: float = 1.15,
    pan_x: float = 0.0,
    pan_y: float = -0.05
) -> Image.Image:
    """
    Crop and resize a square (1080x1080) image to thumbnail (1280x720).

    Uses center crop with zoom and pan offsets for artistic framing.
    pan_x/pan_y: -1.0 to 1.0, 0.0 = centered
    """
    img_w, img_h = image.size

    # Target aspect ratio
    target_ratio = THUMB_WIDTH / THUMB_HEIGHT  # ~1.778

    # Calculate crop region in source image coordinates
    # For a square image, we crop a horizontal band
    crop_w = img_w / zoom
    crop_h = crop_w / target_ratio

    # Ensure crop doesn't exceed image bounds
    crop_h = min(crop_h, img_h / zoom)
    crop_w = crop_h * target_ratio

    # Center + pan offset
    max_offset_x = (img_w - crop_w) / 2.0
    max_offset_y = (img_h - crop_h) / 2.0

    cx = (img_w - crop_w) / 2.0 + pan_x * max_offset_x
    cy = (img_h - crop_h) / 2.0 + pan_y * max_offset_y

    # Clamp to image bounds
    cx = max(0, min(cx, img_w - crop_w))
    cy = max(0, min(cy, img_h - crop_h))

    # Crop and resize
    cropped = image.crop((
        int(cx), int(cy),
        int(cx + crop_w), int(cy + crop_h)
    ))

    return cropped.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.Resampling.LANCZOS)


# ── Overlay Effects ──────────────────────────────────────────────────

def apply_color_wash(image: Image.Image, config: dict) -> Image.Image:
    """Apply a subtle color tint over the entire image."""
    wash_cfg = config.get("color_wash", {})
    if not wash_cfg.get("enabled", False):
        return image

    color = tuple(wash_cfg.get("color", [255, 200, 120]))
    opacity = wash_cfg.get("opacity", 25)

    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (*color, opacity))
    return Image.alpha_composite(rgba, overlay)


def apply_gradient_overlay(image: Image.Image, config: dict) -> Image.Image:
    """
    Apply a gradient overlay for text readability.

    Supports multiple directions:
    - bottom_to_top: dark at bottom, transparent at top (default/classic)
    - top_to_bottom: color at top, transparent at bottom
    - left_to_right: color at left, transparent at right
    """
    grad_cfg = config.get("gradient_overlay", {})
    if not grad_cfg.get("enabled", True):
        return image

    direction = grad_cfg.get("direction", "bottom_to_top")
    start_pct = grad_cfg.get("start_y_percent", 35)
    max_opacity = grad_cfg.get("max_opacity", 225)
    color = tuple(grad_cfg.get("color", [10, 22, 40]))
    curve = grad_cfg.get("curve_exponent", 1.5)

    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = rgba.size

    if direction == "bottom_to_top":
        start_y = int(height * start_pct / 100)
        span = height - start_y
        if span <= 0:
            return image
        for y in range(start_y, height):
            progress = (y - start_y) / span
            alpha = min(int(max_opacity * (progress ** curve)), 255)
            draw.rectangle([0, y, width, y + 1], fill=(*color, alpha))

    elif direction == "top_to_bottom":
        end_y = int(height * (100 - start_pct) / 100)
        if end_y <= 0:
            return image
        for y in range(0, end_y):
            progress = 1.0 - (y / end_y)
            alpha = min(int(max_opacity * (progress ** curve)), 255)
            draw.rectangle([0, y, width, y + 1], fill=(*color, alpha))

    elif direction == "left_to_right":
        end_x = int(width * (100 - start_pct) / 100)
        if end_x <= 0:
            return image
        for x in range(0, end_x):
            progress = 1.0 - (x / end_x)
            alpha = min(int(max_opacity * (progress ** curve)), 255)
            draw.rectangle([x, 0, x + 1, height], fill=(*color, alpha))

    return Image.alpha_composite(rgba, overlay)


def apply_bottom_band(image: Image.Image, config: dict) -> Image.Image:
    """Apply a solid semi-transparent band at the bottom of the image."""
    band_cfg = config.get("bottom_band", {})
    if not band_cfg.get("enabled", False):
        return image

    height_pct = band_cfg.get("height_percent", 20)
    color = tuple(band_cfg.get("color", [255, 255, 255]))
    opacity = band_cfg.get("opacity", 180)

    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = rgba.size
    band_y = int(height * (100 - height_pct) / 100)

    draw.rectangle([0, band_y, width, height], fill=(*color, opacity))

    # Optional top line on the band
    top_line = band_cfg.get("top_line", {})
    if top_line.get("enabled", False):
        line_color = hex_to_rgb(top_line.get("color", "#FFFFFF"))
        line_height = top_line.get("height", 2)
        if len(line_color) == 3:
            line_color = (*line_color, 255)
        draw.rectangle([0, band_y, width, band_y + line_height], fill=line_color)

    return Image.alpha_composite(rgba, overlay)


def apply_vignette(image: Image.Image, config: dict) -> Image.Image:
    """Darken the edges of the image for a cinematic vignette effect."""
    vig_cfg = config.get("vignette", {})
    if not vig_cfg.get("enabled", False):
        return image

    strength = vig_cfg.get("strength", 0.4)
    color = tuple(vig_cfg.get("color", [0, 0, 0]))

    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = rgba.size

    edge_size_x = int(width * 0.2)
    edge_size_y = int(height * 0.2)
    max_alpha = int(255 * strength)

    # Top edge
    for y in range(edge_size_y):
        progress = 1.0 - (y / edge_size_y)
        alpha = int(max_alpha * (progress ** 2))
        draw.rectangle([0, y, width, y + 1], fill=(*color, alpha))

    # Bottom edge
    for y in range(height - edge_size_y, height):
        progress = (y - (height - edge_size_y)) / edge_size_y
        alpha = int(max_alpha * (progress ** 2))
        draw.rectangle([0, y, width, y + 1], fill=(*color, alpha))

    # Left edge
    for x in range(edge_size_x):
        progress = 1.0 - (x / edge_size_x)
        alpha = int(max_alpha * (progress ** 2))
        draw.rectangle([x, 0, x + 1, height], fill=(*color, alpha))

    # Right edge
    for x in range(width - edge_size_x, width):
        progress = (x - (width - edge_size_x)) / edge_size_x
        alpha = int(max_alpha * (progress ** 2))
        draw.rectangle([x, 0, x + 1, height], fill=(*color, alpha))

    return Image.alpha_composite(rgba, overlay)


# ── Drawing Elements ─────────────────────────────────────────────────

def draw_rounded_rectangle(
    draw: ImageDraw.Draw,
    xy: Tuple[int, int, int, int],
    fill: Tuple[int, ...],
    radius: int = 6
) -> None:
    """Draw a rectangle with rounded corners."""
    x1, y1, x2, y2 = xy
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if r <= 0:
        draw.rectangle([x1, y1, x2, y2], fill=fill)
        return

    # Main rectangle minus corners
    draw.rectangle([x1 + r, y1, x2 - r, y2], fill=fill)
    draw.rectangle([x1, y1 + r, x2, y2 - r], fill=fill)

    # Corner circles
    draw.pieslice([x1, y1, x1 + 2*r, y1 + 2*r], 180, 270, fill=fill)
    draw.pieslice([x2 - 2*r, y1, x2, y1 + 2*r], 270, 360, fill=fill)
    draw.pieslice([x1, y2 - 2*r, x1 + 2*r, y2], 90, 180, fill=fill)
    draw.pieslice([x2 - 2*r, y2 - 2*r, x2, y2], 0, 90, fill=fill)


def wrap_title_lines(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.Draw,
    max_lines: int = 3
) -> List[str]:
    """
    Wrap title text into lines that fit within max_width.
    Respects explicit newlines (\\n) in the text as forced line breaks.

    Returns up to max_lines lines. If text exceeds max_lines,
    the last line is truncated with '...'.
    """
    # Split by explicit newlines first, then word-wrap each segment
    segments = text.split('\n')
    lines = []

    for segment in segments:
        if len(lines) >= max_lines:
            break

        words = segment.split()
        if not words:
            continue

        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                if len(lines) >= max_lines:
                    break

        if current_line and len(lines) < max_lines:
            lines.append(' '.join(current_line))

    return lines[:max_lines]


def draw_title_text(
    image: Image.Image,
    title: str,
    config: dict
) -> Tuple[Image.Image, int, int, int]:
    """
    Draw the title text with word-level highlighting on UPPERCASE words.

    Supports left, center, and right alignment.

    Returns (image, title_top_y, title_bottom_y, title_center_x) for accent
    element positioning.
    """
    title_cfg = config.get("title", {})
    font_name = title_cfg.get("font", "Montserrat-ExtraBold.ttf")
    font_size = title_cfg.get("font_size", 82)
    text_color = hex_to_rgb(title_cfg.get("color", "#FFFFFF"))
    highlight_color = hex_to_rgb(title_cfg.get("highlight_color", "#FFD700"))
    stroke_width = title_cfg.get("stroke_width", 5)
    stroke_color = hex_to_rgb(title_cfg.get("stroke_color", "#0A1628"))
    shadow_offset = tuple(title_cfg.get("shadow_offset", [3, 3]))
    shadow_color = hex_to_rgb(title_cfg.get("shadow_color", "#00000080"))
    max_width_pct = title_cfg.get("max_width_percent", 85)
    pos_y_pct = title_cfg.get("position_y_percent", 62)
    margin_left = title_cfg.get("margin_left", 60)
    line_spacing = title_cfg.get("line_spacing", 14)
    max_lines = title_cfg.get("max_lines", 3)
    align = title_cfg.get("align", "left")

    # Use custom text if provided
    custom_text = title_cfg.get("custom_text")
    if custom_text and title_cfg.get("text") == "custom":
        title = custom_text

    font = get_font(font_name, font_size)
    rgba = image.convert("RGBA")
    txt_layer = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    max_width = int(THUMB_WIDTH * max_width_pct / 100) - margin_left

    # Wrap text
    lines = wrap_title_lines(title, font, max_width, draw, max_lines)

    # Calculate line height
    ref_bbox = draw.textbbox((0, 0), "Mg", font=font)
    line_height = ref_bbox[3] - ref_bbox[1]

    # Position: start_y based on percent
    start_y = int(THUMB_HEIGHT * pos_y_pct / 100)

    # Track title bounds for accent element positioning
    title_top_y = start_y
    title_bottom_y = start_y
    title_center_x = THUMB_WIDTH // 2

    # Draw each line with word-level highlighting
    for i, line in enumerate(lines):
        y = start_y + i * (line_height + line_spacing)
        title_bottom_y = y + line_height

        words = line.split(' ')

        # Calculate line width for alignment
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]

        if align == "center":
            current_x = (THUMB_WIDTH - line_width) // 2
        elif align == "right":
            margin_right = title_cfg.get("margin_right", 60)
            current_x = THUMB_WIDTH - line_width - margin_right
        else:  # left
            current_x = margin_left

        for wi, word in enumerate(words):
            # Check if word is highlighted (UPPERCASE, 2+ chars)
            clean = word.replace('¿', '').replace('?', '').replace('¡', '')
            clean = clean.replace('!', '').replace(',', '').replace('.', '')
            clean = clean.replace(':', '').replace(';', '')
            is_highlight = len(clean) >= 2 and clean.isupper()
            color = highlight_color if is_highlight else text_color

            display_word = word + (' ' if wi < len(words) - 1 else '')

            # Shadow layer
            sx, sy = shadow_offset
            if sx != 0 or sy != 0:
                draw.text(
                    (current_x + sx, y + sy), display_word,
                    font=font, fill=shadow_color
                )

            # Stroke + main text
            draw.text(
                (current_x, y), display_word,
                font=font, fill=color,
                stroke_width=stroke_width, stroke_fill=stroke_color
            )

            # Advance x
            wbbox = draw.textbbox((0, 0), display_word, font=font)
            current_x += wbbox[2] - wbbox[0]

    result = Image.alpha_composite(rgba, txt_layer)
    return result, title_top_y, title_bottom_y, title_center_x


# ── Accent Elements ──────────────────────────────────────────────────

def draw_accent_element(
    image: Image.Image,
    title_config: dict,
    config: dict,
    title_top_y: int,
    title_bottom_y: int
) -> Image.Image:
    """
    Draw accent elements based on type. Supports multiple styles.

    Falls back to legacy accent_line config if accent_element is not present.
    """
    # Check for new accent_element config first
    accent_cfg = config.get("accent_element", {})

    # Fall back to legacy accent_line if accent_element not present
    if not accent_cfg:
        accent_cfg = config.get("accent_line", {})
        if accent_cfg.get("enabled", False):
            accent_cfg["type"] = "line_above"
        else:
            return image

    if not accent_cfg.get("enabled", True):
        return image

    element_type = accent_cfg.get("type", "line_above")

    if element_type == "line_above":
        return _draw_line_above(image, title_config, accent_cfg, title_top_y)
    elif element_type == "divider_below":
        return _draw_divider_below(image, title_config, accent_cfg, title_bottom_y)
    elif element_type == "frame":
        return _draw_frame(image, title_config, accent_cfg, title_top_y, title_bottom_y)
    elif element_type == "vertical_line":
        return _draw_vertical_line(image, accent_cfg)
    elif element_type == "diamond":
        return _draw_diamond(image, title_config, accent_cfg, title_top_y, title_bottom_y)
    return image


def _draw_line_above(
    image: Image.Image, title_cfg: dict, accent_cfg: dict, title_top_y: int
) -> Image.Image:
    """Draw a horizontal accent line above the title."""
    color = hex_to_rgb(accent_cfg.get("color", "#FFD700"))
    height = accent_cfg.get("height", 4)
    margin_left = accent_cfg.get("margin_left", 60)
    width_pct = accent_cfg.get("width_percent", 25)
    margin_below = accent_cfg.get("margin_bottom_from_title", 12)

    line_y = title_top_y - margin_below - height
    line_width = int(THUMB_WIDTH * width_pct / 100)

    align = title_cfg.get("align", "left")
    if align == "center":
        line_x = (THUMB_WIDTH - line_width) // 2
    else:
        line_x = margin_left

    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    draw.rectangle(
        [line_x, line_y, line_x + line_width, line_y + height],
        fill=color
    )
    return rgba


def _draw_divider_below(
    image: Image.Image, title_cfg: dict, accent_cfg: dict, title_bottom_y: int
) -> Image.Image:
    """Draw a horizontal divider line below the title."""
    color = hex_to_rgb(accent_cfg.get("color", "#B8860B"))
    height = accent_cfg.get("height", 2)
    width_pct = accent_cfg.get("width_percent", 30)
    margin_from_title = accent_cfg.get("margin_from_title", 12)

    line_width = int(THUMB_WIDTH * width_pct / 100)
    line_y = title_bottom_y + margin_from_title

    align = title_cfg.get("align", "left")
    if align == "center":
        line_x = (THUMB_WIDTH - line_width) // 2
    else:
        margin_left = title_cfg.get("margin_left", 60)
        line_x = margin_left

    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    draw.rectangle(
        [line_x, line_y, line_x + line_width, line_y + height],
        fill=color
    )
    return rgba


def _draw_frame(
    image: Image.Image, title_cfg: dict, accent_cfg: dict,
    title_top_y: int, title_bottom_y: int
) -> Image.Image:
    """Draw two lines (above and below) the title as an elegant frame."""
    color = hex_to_rgb(accent_cfg.get("color", "#FFD700"))
    height = accent_cfg.get("height", 2)
    width_pct = accent_cfg.get("width_percent", 45)
    margin = accent_cfg.get("margin_from_title", 16)

    line_width = int(THUMB_WIDTH * width_pct / 100)

    align = title_cfg.get("align", "left")
    if align == "center":
        line_x = (THUMB_WIDTH - line_width) // 2
    else:
        margin_left = title_cfg.get("margin_left", 60)
        line_x = margin_left

    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)

    # Line above
    top_y = title_top_y - margin - height
    draw.rectangle([line_x, top_y, line_x + line_width, top_y + height], fill=color)

    # Line below
    bot_y = title_bottom_y + margin
    draw.rectangle([line_x, bot_y, line_x + line_width, bot_y + height], fill=color)

    return rgba


def _draw_vertical_line(image: Image.Image, accent_cfg: dict) -> Image.Image:
    """Draw a vertical accent bar on the left side."""
    color = hex_to_rgb(accent_cfg.get("color", "#F0A030"))
    width = accent_cfg.get("width", 4)
    margin_left = accent_cfg.get("margin_left", 40)
    start_y_pct = accent_cfg.get("start_y_percent", 25)
    height_pct = accent_cfg.get("height_percent", 50)

    start_y = int(THUMB_HEIGHT * start_y_pct / 100)
    line_height = int(THUMB_HEIGHT * height_pct / 100)

    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    draw.rectangle(
        [margin_left, start_y, margin_left + width, start_y + line_height],
        fill=color
    )
    return rgba


def _draw_diamond(
    image: Image.Image, title_cfg: dict, accent_cfg: dict,
    title_top_y: int, title_bottom_y: int
) -> Image.Image:
    """Draw a small rotated diamond shape next to the title."""
    color = hex_to_rgb(accent_cfg.get("color", "#FFD700"))
    size = accent_cfg.get("size", 18)
    margin_left = accent_cfg.get("margin_left", 60)

    # Position diamond to the left of and vertically centered on the title
    cx = margin_left - size - 10
    cy = (title_top_y + title_bottom_y) // 2

    # Diamond is a rotated square
    points = [
        (cx, cy - size),      # top
        (cx + size, cy),      # right
        (cx, cy + size),      # bottom
        (cx - size, cy),      # left
    ]

    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    draw.polygon(points, fill=color)
    return rgba


# ── Legacy accent_line (backwards compatibility) ─────────────────────

def draw_accent_line(
    image: Image.Image,
    title_config: dict,
    accent_config: dict
) -> Image.Image:
    """Draw a gold accent line above the title area. Legacy function."""
    if not accent_config.get("enabled", True):
        return image

    color = hex_to_rgb(accent_config.get("color", "#FFD700"))
    height = accent_config.get("height", 4)
    margin_left = accent_config.get("margin_left", 60)
    width_pct = accent_config.get("width_percent", 25)
    margin_below = accent_config.get("margin_bottom_from_title", 12)

    # Position above the title
    pos_y_pct = title_config.get("position_y_percent", 62)
    title_y = int(THUMB_HEIGHT * pos_y_pct / 100)
    line_y = title_y - margin_below - height
    line_width = int(THUMB_WIDTH * width_pct / 100)

    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    draw.rectangle(
        [margin_left, line_y, margin_left + line_width, line_y + height],
        fill=color
    )

    return rgba


# ── Badges ───────────────────────────────────────────────────────────

def draw_scripture_badge(
    image: Image.Image,
    escritura: str,
    config: dict
) -> Image.Image:
    """Draw a scripture reference badge. Supports left, center, right alignment."""
    badge_cfg = config.get("scripture_badge", {})
    if not badge_cfg.get("enabled", True) or not escritura:
        return image

    # Use custom text if provided
    custom_text = badge_cfg.get("custom_text")
    if custom_text and badge_cfg.get("text") == "custom":
        escritura = custom_text

    font_name = badge_cfg.get("font", "Montserrat-Bold.ttf")
    font_size = badge_cfg.get("font_size", 32)
    text_color = hex_to_rgb(badge_cfg.get("text_color", "#FFFFFF"))
    bg_color = hex_to_rgb(badge_cfg.get("background_color", "#E87722"))
    pad_h = badge_cfg.get("padding_h", 20)
    pad_v = badge_cfg.get("padding_v", 8)
    margin_left = badge_cfg.get("margin_left", 60)
    margin_bottom = badge_cfg.get("margin_bottom", 40)
    margin_right = badge_cfg.get("margin_right", 40)
    radius = badge_cfg.get("corner_radius", 6)
    align = badge_cfg.get("align", "left")

    font = get_font(font_name, font_size)
    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)

    # Measure text
    bbox = draw.textbbox((0, 0), escritura, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    text_offset_y = bbox[1]  # vertical offset from textbbox origin

    badge_w = tw + 2 * pad_h
    badge_h = th + 2 * pad_v

    by2 = THUMB_HEIGHT - margin_bottom
    by1 = by2 - badge_h

    if align == "center":
        bx1 = (THUMB_WIDTH - badge_w) // 2
    elif align == "right":
        bx1 = THUMB_WIDTH - badge_w - margin_right
    else:  # left
        bx1 = margin_left

    bx2 = bx1 + badge_w

    draw_rounded_rectangle(draw, (bx1, by1, bx2, by2), fill=bg_color, radius=radius)

    # Center text in badge (account for textbbox offset to truly center)
    tx = bx1 + pad_h
    ty = by1 + (badge_h - th) // 2 - text_offset_y
    draw.text((tx, ty), escritura, font=font, fill=text_color)

    return rgba


def draw_branding_badge(
    image: Image.Image,
    config: dict
) -> Image.Image:
    """
    Draw the 'VEN SÍGUEME' branding badge.
    Supports positions: bottom_right, top_right, top_left, bottom_left.
    """
    brand_cfg = config.get("branding", {})
    if not brand_cfg.get("enabled", True):
        return image

    text = brand_cfg.get("text", "VEN SÍGUEME")
    font_name = brand_cfg.get("font", "Montserrat-Bold.ttf")
    font_size = brand_cfg.get("font_size", 26)
    text_color = hex_to_rgb(brand_cfg.get("text_color", "#FFFFFF"))
    bg_color = hex_to_rgb(brand_cfg.get("background_color", "#005DA6"))
    pad_h = brand_cfg.get("padding_h", 16)
    pad_v = brand_cfg.get("padding_v", 6)
    margin_right = brand_cfg.get("margin_right", 40)
    margin_bottom = brand_cfg.get("margin_bottom", 40)
    margin_left = brand_cfg.get("margin_left", 40)
    margin_top = brand_cfg.get("margin_top", 30)
    radius = brand_cfg.get("corner_radius", 6)
    position = brand_cfg.get("position", "bottom_right")

    font = get_font(font_name, font_size)
    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    badge_w = tw + 2 * pad_h
    badge_h = th + 2 * pad_v

    if position == "bottom_right":
        bx2 = THUMB_WIDTH - margin_right
        bx1 = bx2 - badge_w
        by2 = THUMB_HEIGHT - margin_bottom
        by1 = by2 - badge_h
    elif position == "top_right":
        bx2 = THUMB_WIDTH - margin_right
        bx1 = bx2 - badge_w
        by1 = margin_top
        by2 = by1 + badge_h
    elif position == "top_left":
        bx1 = margin_left
        bx2 = bx1 + badge_w
        by1 = margin_top
        by2 = by1 + badge_h
    elif position == "bottom_left":
        bx1 = margin_left
        bx2 = bx1 + badge_w
        by2 = THUMB_HEIGHT - margin_bottom
        by1 = by2 - badge_h
    else:  # default bottom_right
        bx2 = THUMB_WIDTH - margin_right
        bx1 = bx2 - badge_w
        by2 = THUMB_HEIGHT - margin_bottom
        by1 = by2 - badge_h

    draw_rounded_rectangle(draw, (bx1, by1, bx2, by2), fill=bg_color, radius=radius)

    # Center text in badge
    tx = bx1 + pad_h
    ty = by1 + pad_v - 2
    draw.text((tx, ty), text, font=font, fill=text_color)

    return rgba


# ── Main Pipeline ────────────────────────────────────────────────────

def generate_thumbnail(
    output_path: Optional[Path] = None,
    title_override: Optional[str] = None,
    image_path_override: Optional[Path] = None
) -> Path:
    """
    Generate a professional YouTube thumbnail.

    Pipeline:
    1. Load image (from video assets or custom)
    2. Crop to 1280x720 with zoom/pan
    3. Apply color wash (if enabled)
    4. Apply gradient overlay
    5. Apply bottom band (if enabled)
    6. Apply vignette (if enabled)
    7. Draw title with CAPS highlights
    8. Draw accent element
    9. Draw scripture badge
    10. Draw branding badge
    11. Save as PNG

    Returns:
        Path to the generated thumbnail
    """
    print("=" * 50)
    print("THUMBNAIL GENERATOR")
    print("=" * 50)

    # Load configs
    config = load_config()
    thumb_full = load_thumbnail_config()
    thumb_cfg = thumb_full.get("thumbnail", {})
    script = load_script()

    # Get metadata
    script_data = script.get("script", {})
    title_cfg = thumb_cfg.get("title", {})

    # Title priority: CLI override > custom_text in config > title_youtube > topic
    if title_override:
        title = title_override
    elif title_cfg.get("text") == "custom" and title_cfg.get("custom_text"):
        title = title_cfg["custom_text"]
    else:
        title = script_data.get("title_youtube", script_data.get("topic", "Ven Sígueme"))

    metadata = config.get("video_metadata", {})
    escritura = metadata.get("escritura", "")

    # Check for layout name in config (for logging)
    layout_name = thumb_cfg.get("_layout_name", "custom")
    print(f"Layout: {layout_name}")
    print(f"Title: {title}")
    print(f"Scripture: {escritura}")

    # 1. Find and load image
    if image_path_override:
        img_path = image_path_override
    else:
        img_path = find_best_image(script, thumb_cfg)

    if not img_path or not img_path.exists():
        raise FileNotFoundError(
            "No image found for thumbnail. "
            "Generate images first or specify custom_path in thumbnail_config.json"
        )

    print(f"Image: {img_path.name}")
    source_img = Image.open(img_path).convert("RGBA")

    # 2. Crop to thumbnail ratio
    image_cfg = thumb_cfg.get("image", {})
    thumb = crop_to_thumbnail(
        source_img,
        zoom=image_cfg.get("zoom", 1.15),
        pan_x=image_cfg.get("pan_x", 0.0),
        pan_y=image_cfg.get("pan_y", -0.05)
    )
    print(f"Cropped to {THUMB_WIDTH}x{THUMB_HEIGHT}")

    # 3. Color wash (subtle tint over entire image)
    thumb = apply_color_wash(thumb, thumb_cfg)

    # 4. Gradient overlay
    thumb = apply_gradient_overlay(thumb, thumb_cfg)

    # 5. Bottom band (solid color block)
    thumb = apply_bottom_band(thumb, thumb_cfg)

    # 6. Vignette (darken edges)
    thumb = apply_vignette(thumb, thumb_cfg)

    # 7. Title text (returns image + position info for accent elements)
    title_cfg = thumb_cfg.get("title", {})
    thumb, title_top_y, title_bottom_y, title_center_x = draw_title_text(
        thumb, title, thumb_cfg
    )

    # 8. Accent element (uses title position for alignment)
    # Check if new accent_element system is configured; otherwise use legacy accent_line
    has_accent_element = "accent_element" in thumb_cfg
    has_legacy_accent = "accent_line" in thumb_cfg

    if has_accent_element:
        thumb = draw_accent_element(
            thumb, title_cfg, thumb_cfg, title_top_y, title_bottom_y
        )
    elif has_legacy_accent:
        accent_cfg = thumb_cfg.get("accent_line", {})
        thumb = draw_accent_line(thumb, title_cfg, accent_cfg)

    # 9. Scripture badge
    thumb = draw_scripture_badge(thumb, escritura, thumb_cfg)

    # 10. Branding badge
    thumb = draw_branding_badge(thumb, thumb_cfg)

    # 11. Save
    if output_path is None:
        out_str = thumb_cfg.get("output_path", "data/output/thumbnail.png")
        output_path = BASE_DIR / out_str

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGBA → RGB for PNG without transparency
    final = thumb.convert("RGB")
    final.save(str(output_path), "PNG", quality=95)

    print(f"\nThumbnail saved: {output_path}")
    print(f"Size: {THUMB_WIDTH}x{THUMB_HEIGHT}")
    print("=" * 50)

    return output_path


def main():
    """Entry point for thumbnail generation."""
    generate_thumbnail()


if __name__ == "__main__":
    main()
