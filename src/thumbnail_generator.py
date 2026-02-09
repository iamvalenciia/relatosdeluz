"""
Thumbnail Generator - Professional YouTube thumbnails for LDS Ven Sígueme content.

Design principles (research-backed for 45-65+ Spanish-speaking LDS audience):
- Full-bleed sacred art image with dark navy gradient overlay (bottom 35-45%)
- Large Montserrat ExtraBold text (72-90px), UPPERCASE words highlighted in gold
- Scripture badge (orange) + "VEN SÍGUEME" brand badge (blue)
- Gold accent line above title for visual separation
- High contrast (7:1+) for older demographic readability
- 3-5 words max in title for CTR optimization

Fonts recommended (download from Google Fonts → data/fonts/):
  - Montserrat-ExtraBold.ttf (primary headline)
  - Montserrat-Bold.ttf (badges, secondary text)
  - Oswald-SemiBold.ttf (alternative for longer Spanish text)
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


def apply_gradient_overlay(
    image: Image.Image,
    config: dict
) -> Image.Image:
    """
    Apply a bottom-to-transparent gradient overlay for text readability.

    Uses a dark navy blue gradient with configurable opacity and curve.
    """
    grad_cfg = config.get("gradient_overlay", {})
    if not grad_cfg.get("enabled", True):
        return image

    start_y_pct = grad_cfg.get("start_y_percent", 35)
    max_opacity = grad_cfg.get("max_opacity", 225)
    color = tuple(grad_cfg.get("color", [10, 22, 40]))
    curve = grad_cfg.get("curve_exponent", 1.5)

    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = rgba.size
    start_y = int(height * start_y_pct / 100)

    for y in range(start_y, height):
        progress = (y - start_y) / (height - start_y)
        alpha = int(max_opacity * (progress ** curve))
        alpha = min(alpha, 255)
        draw.rectangle([0, y, width, y + 1], fill=(*color, alpha))

    composited = Image.alpha_composite(rgba, overlay)
    return composited


def draw_rounded_rectangle(
    draw: ImageDraw.Draw,
    xy: Tuple[int, int, int, int],
    fill: Tuple[int, ...],
    radius: int = 6
) -> None:
    """Draw a rectangle with rounded corners."""
    x1, y1, x2, y2 = xy
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

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
) -> Image.Image:
    """
    Draw the title text with word-level gold highlighting on UPPERCASE words.

    Features:
    - Large bold font (Montserrat ExtraBold)
    - White text with gold highlights on CAPS words
    - Dark stroke outline for readability
    - Subtle drop shadow
    - Left-aligned, positioned in lower portion over gradient
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
    total_text_height = line_height * len(lines) + line_spacing * (len(lines) - 1)
    start_y = int(THUMB_HEIGHT * pos_y_pct / 100)

    # Draw each line with word-level highlighting
    for i, line in enumerate(lines):
        y = start_y + i * (line_height + line_spacing)
        words = line.split(' ')
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

    return Image.alpha_composite(rgba, txt_layer)


def draw_accent_line(
    image: Image.Image,
    title_config: dict,
    accent_config: dict
) -> Image.Image:
    """Draw a gold accent line above the title area."""
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


def draw_scripture_badge(
    image: Image.Image,
    escritura: str,
    config: dict
) -> Image.Image:
    """
    Draw a scripture reference badge (orange pill) at the bottom left.
    """
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
    radius = badge_cfg.get("corner_radius", 6)

    font = get_font(font_name, font_size)
    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)

    # Measure text
    bbox = draw.textbbox((0, 0), escritura, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Badge position (bottom-left)
    bx1 = margin_left
    by2 = THUMB_HEIGHT - margin_bottom
    by1 = by2 - th - 2 * pad_v
    bx2 = bx1 + tw + 2 * pad_h

    draw_rounded_rectangle(draw, (bx1, by1, bx2, by2), fill=bg_color, radius=radius)

    # Center text in badge
    tx = bx1 + pad_h
    ty = by1 + pad_v - 2  # slight visual adjustment
    draw.text((tx, ty), escritura, font=font, fill=text_color)

    return rgba


def draw_branding_badge(
    image: Image.Image,
    config: dict
) -> Image.Image:
    """
    Draw the 'VEN SÍGUEME' branding badge (blue pill) at bottom right.
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
    radius = brand_cfg.get("corner_radius", 6)

    font = get_font(font_name, font_size)
    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Badge position (bottom-right)
    bx2 = THUMB_WIDTH - margin_right
    bx1 = bx2 - tw - 2 * pad_h
    by2 = THUMB_HEIGHT - margin_bottom
    by1 = by2 - th - 2 * pad_v

    draw_rounded_rectangle(draw, (bx1, by1, bx2, by2), fill=bg_color, radius=radius)

    # Center text in badge
    tx = bx1 + pad_h
    ty = by1 + pad_v - 2
    draw.text((tx, ty), text, font=font, fill=text_color)

    return rgba


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
    3. Apply gradient overlay
    4. Draw gold accent line
    5. Draw title with CAPS highlights
    6. Draw scripture badge (orange)
    7. Draw branding badge (blue)
    8. Save as PNG

    Args:
        output_path: Override output file path
        title_override: Override title text
        image_path_override: Override source image

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

    # 3. Gradient overlay
    thumb = apply_gradient_overlay(thumb, thumb_cfg)

    # 4. Accent line
    title_cfg = thumb_cfg.get("title", {})
    accent_cfg = thumb_cfg.get("accent_line", {})
    thumb = draw_accent_line(thumb, title_cfg, accent_cfg)

    # 5. Title text
    thumb = draw_title_text(thumb, title, thumb_cfg)

    # 6. Scripture badge
    thumb = draw_scripture_badge(thumb, escritura, thumb_cfg)

    # 7. Branding badge
    thumb = draw_branding_badge(thumb, thumb_cfg)

    # 8. Save
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
