"""
Video Renderer - Hybrid rendering engine using PyAV + Pillow + NumPy
Uses NVIDIA GPU acceleration (hevc_nvenc) for RTX 3060 TI.

Features:
- Ken Burns effect (zoom + pan)
- Dual format: Horizontal 1920x1080 (16:9) + Vertical 1080x1920 (9:16)
- 1:1 square content images centered over blurred decorative background
- Professional TV news lower third overlay (horizontal)
- Title text above images (vertical)
- Opening sweep animation
- Image sync with audio timestamps
"""

import os
import json
import math
import time as _time_module
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import av
from moviepy.editor import AudioFileClip

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
MUSIC_DIR = DATA_DIR / "music"
OUTPUT_DIR = DATA_DIR / "output"
SCRIPTS_DIR = DATA_DIR / "scripts"
CONFIG_PATH = DATA_DIR / "config.json"
RENDER_PROGRESS_PATH = DATA_DIR / ".render_progress.json"


def update_render_progress(phase: str, percent: float, detail: str = ""):
    """Write render progress to a JSON file so MCP can report it."""
    progress = {
        "phase": phase,
        "percent": round(percent, 1),
        "detail": detail,
        "timestamp": _time_module.time(),
        "pid": os.getpid()
    }
    try:
        RENDER_PROGRESS_PATH.write_text(
            json.dumps(progress), encoding="utf-8"
        )
    except OSError:
        pass  # Non-critical, don't crash render


def clear_render_progress():
    """Remove the progress file when render is done."""
    RENDER_PROGRESS_PATH.unlink(missing_ok=True)

# Video constants
FPS = 30

# Format dimensions
HORIZONTAL_WIDTH = 1920
HORIZONTAL_HEIGHT = 1080
VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920

# Legacy aliases for backward compatibility
VIDEO_WIDTH = HORIZONTAL_WIDTH
VIDEO_HEIGHT = HORIZONTAL_HEIGHT


def load_config() -> dict:
    """Load configuration from config.json"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_script() -> dict:
    """Load the current script"""
    script_path = SCRIPTS_DIR / "current.json"
    # Read with utf-8-sig to automatically handle BOM
    with open(script_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    # Handle case where Claude CLI adds extra text before/after JSON
    # Find the JSON object boundaries
    start_idx = content.find('{')
    end_idx = content.rfind('}')

    if start_idx == -1 or end_idx == -1:
        raise ValueError("No valid JSON object found in script file")

    json_content = content[start_idx:end_idx + 1]
    return json.loads(json_content)


def load_timestamps() -> dict:
    """Load word-level timestamps"""
    timestamps_path = AUDIO_DIR / "current_timestamps.json"
    if not timestamps_path.exists():
        raise FileNotFoundError(
            f"Timestamps not found at {timestamps_path}. "
            "Run audio_generator.py first."
        )
    with open(timestamps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image(visual_asset_id: str) -> Optional[Path]:
    """Find image file by visual_asset_id regardless of extension"""
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path = IMAGES_DIR / f"{visual_asset_id}{ext}"
        if path.exists():
            return path
    return None


def _time_from_word_index(word_index: int, whisper_words: list, field: str = "start") -> float:
    """Legacy helper: look up a time from a word index in Whisper words."""
    for w in whisper_words:
        if w.get("index", -1) == word_index:
            return w.get(field, 0.0)
    return 0.0


def load_images(
    visual_assets: List[dict],
    timestamps_words: list = None,
) -> List[Tuple[str, Image.Image, float, float]]:
    """
    Load all images referenced in visual_assets.
    Images are loaded at their native resolution (1:1 square).
    Placement on the video canvas is handled by create_frame().

    Returns:
        List of (asset_id, PIL Image, start_time, end_time)
    """
    images = []
    for asset in visual_assets:
        asset_id = asset.get("visual_asset_id")
        img_path = find_image(asset_id)

        if img_path:
            img = Image.open(img_path).convert("RGB")
            # Images stay at native size — resizing to canvas happens in create_frame()

            # Prefer direct time ranges (set by content-based aligner)
            if "start_time" in asset and "end_time" in asset:
                start_t = float(asset["start_time"])
                end_t = float(asset["end_time"])
            elif timestamps_words:
                # Legacy fallback: compute from word indices
                start_t = _time_from_word_index(asset.get("start_word_index", 0), timestamps_words, "start")
                end_t = _time_from_word_index(asset.get("end_word_index", 0), timestamps_words, "end")
            else:
                start_t = 0.0
                end_t = 0.0

            images.append((asset_id, img, start_t, end_t))
            print(f"Loaded image: {asset_id} ({img_path.name}) {img.size[0]}x{img.size[1]} [{start_t:.1f}s - {end_t:.1f}s]")
        else:
            print(f"WARNING: Image not found for asset_id: {asset_id}")

    return images


def load_background_image(
    canvas_width: int,
    canvas_height: int,
    visual_assets: List[dict] = None
) -> Optional[Image.Image]:
    """
    Load a decorative background image, resize to cover the canvas,
    and apply a strong Gaussian blur + slight darkening.

    Priority:
    1. bg.png (dedicated background image)
    2. First content image from visual_assets (fallback)
    3. None (solid black fallback)
    """
    bg_source = None
    source_label = "none"

    # Try dedicated bg.png first
    bg_path = find_image("bg")
    if bg_path is not None:
        bg_source = Image.open(bg_path).convert("RGB")
        source_label = "bg.png"
    elif visual_assets:
        # Fallback: use the first available content image
        for asset in visual_assets:
            asset_id = asset.get("visual_asset_id", "")
            img_path = find_image(asset_id)
            if img_path:
                bg_source = Image.open(img_path).convert("RGB")
                source_label = f"content image {asset_id}"
                break

    if bg_source is None:
        print("WARNING: No background image found. Using solid black.")
        return None

    # Resize to cover the entire canvas (crop if needed)
    bg = resize_cover(bg_source, canvas_width, canvas_height)
    # Apply strong Gaussian blur for the decorative background effect
    bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
    # Slightly darken to ensure content images pop
    bg = ImageEnhance.Brightness(bg).enhance(0.6)
    print(f"Background loaded from {source_label} and blurred: {canvas_width}x{canvas_height}")
    return bg


def resize_cover(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Resize image to cover target dimensions, cropping if necessary.
    This ensures no black bars appear.
    """
    img_w, img_h = image.size
    target_ratio = target_width / target_height
    img_ratio = img_w / img_h

    if img_ratio > target_ratio:
        # Image is wider, fit to height and crop width
        new_height = target_height
        new_width = int(img_w * (target_height / img_h))
    else:
        # Image is taller, fit to width and crop height
        new_width = target_width
        new_height = int(img_h * (target_width / img_w))

    # Resize
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    cropped = resized.crop((left, top, left + target_width, top + target_height))

    return cropped


def get_font(size: int, weight: str = "bold") -> ImageFont.FreeTypeFont:
    """
    Get Montserrat font at the specified size and weight.

    Args:
        size: Font size in pixels
        weight: 'bold', 'semibold', 'medium', or 'extrabold'
    """
    fonts_dir = DATA_DIR / "fonts"

    weight_map = {
        "extrabold": [
            fonts_dir / "Montserrat-ExtraBold.ttf",
            fonts_dir / "Montserrat-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ],
        "bold": [
            fonts_dir / "Montserrat-Bold.ttf",
            fonts_dir / "EBGaramond-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ],
        "semibold": [
            fonts_dir / "Montserrat-SemiBold.ttf",
            fonts_dir / "Montserrat-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ],
        "medium": [
            fonts_dir / "Montserrat-Medium.ttf",
            fonts_dir / "Montserrat-SemiBold.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ],
    }

    font_list = weight_map.get(weight, weight_map["bold"])

    for font_path in font_list:
        path = str(font_path) if hasattr(font_path, 'exists') else font_path
        try:
            if (hasattr(font_path, 'exists') and font_path.exists()) or \
               (isinstance(font_path, str) and os.path.exists(font_path)):
                return ImageFont.truetype(path, size)
        except:
            continue

    return ImageFont.load_default()


# Keep backward compatibility alias
get_title_font = get_font


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def calculate_image_placement(
    canvas_width: int,
    canvas_height: int,
    video_format: str
) -> Tuple[int, int, int, int]:
    """
    Calculate the position and size of the square content image on the canvas.

    Returns (x, y, target_width, target_height) for where to paste the image.
    """
    if video_format == "vertical":
        # Vertical (1080x1920): image fills width, centered vertically
        # Layout: [title above] [image] [tags below]
        # Title area: ~220px top, Tags area: ~120px bottom
        TITLE_AREA = 220
        TAGS_AREA = 120
        target_size = canvas_width  # 1080px wide = full width
        x = 0
        # Center the image in the remaining space between title and tags
        available_height = canvas_height - TITLE_AREA - TAGS_AREA
        y = TITLE_AREA + (available_height - target_size) // 2
        return x, y, target_size, target_size
    else:
        # Horizontal (1920x1080): image fits height, centered horizontally
        target_size = canvas_height  # 1080px tall = full height
        x = (canvas_width - target_size) // 2  # Center horizontally (~420px from each side)
        y = 0
        return x, y, target_size, target_size


def apply_ken_burns(
    image: Image.Image,
    progress: float,
    target_w: int,
    target_h: int,
    zoom_start: float = 1.0,
    zoom_end: float = 1.08,
    pan_direction: str = "center"
) -> Image.Image:
    """
    Apply Ken Burns effect using affine transform for sub-pixel smooth animation.

    Uses PIL's Image.transform with AFFINE mode to avoid integer rounding artifacts
    that cause visible jitter/shaking. All coordinates stay in floating-point space
    until the final bicubic interpolation handles sub-pixel positioning natively.

    Args:
        image: Source image (any size, typically 1:1 square)
        progress: Animation progress (0.0 to 1.0)
        target_w: Output width
        target_h: Output height
        zoom_start: Initial zoom level
        zoom_end: Final zoom level
        pan_direction: Direction of pan ("center", "left", "right")

    Returns:
        Transformed image at target_w x target_h
    """
    # Smooth easing function (ease-in-out cosine)
    ease_progress = 0.5 - 0.5 * math.cos(progress * math.pi)

    # Current zoom level (fully floating-point, no rounding)
    current_zoom = zoom_start + (zoom_end - zoom_start) * ease_progress

    img_w, img_h = image.size

    # The visible region in source-image coordinates:
    # At zoom=1.0 we see the full image; at zoom=1.08 we see a smaller crop (zoomed in).
    crop_w = img_w / current_zoom
    crop_h = img_h / current_zoom

    # Center the crop region by default
    cx = (img_w - crop_w) / 2.0
    cy = (img_h - crop_h) / 2.0

    # Apply panning offset (shift within the available margin)
    max_pan_x = (img_w - crop_w) / 2.0
    if pan_direction == "left":
        cx += max_pan_x * (1.0 - ease_progress)
    elif pan_direction == "right":
        cx -= max_pan_x * (1.0 - ease_progress)

    # Build affine transform coefficients
    scale_x = crop_w / target_w
    scale_y = crop_h / target_h

    affine_coeffs = (
        scale_x, 0.0, cx,
        0.0, scale_y, cy,
    )

    # Convert to RGB before transform - RGBA affine transforms are extremely slow
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # BICUBIC resampling gives smooth sub-pixel interpolation with no jitter
    result = image.transform(
        (target_w, target_h),
        Image.AFFINE,
        affine_coeffs,
        resample=Image.Resampling.BICUBIC,
    )

    return result


def draw_professional_lower_third(
    draw: ImageDraw.Draw,
    frame: Image.Image,
    title: str,
    frame_num: int,
    total_frames: int,
    config: dict,
    canvas_width: int = 1920,
    canvas_height: int = 1080
) -> None:
    """
    Draw a professional news-style lower third at the bottom left.
    Features an opening animation and smart text truncation.
    """
    # Colors (LDS Branding)
    BADGE_BG = (242, 169, 0, 255)           # Orange/Gold for badge
    TITLE_BG = (255, 255, 255, 230)        # Semi-transparent white
    SCRIPTURE_BG = (0, 46, 93, 240)        # Deep navy blue for scriptures
    TEXT_COLOR = (20, 20, 20, 255)          # Near black for title

    # Metadata from config
    metadata = config.get("video_metadata", {})
    programa = metadata.get("programa", "Ven Sígueme")
    escritura = metadata.get("escritura", "")

    # Format scripture: add separator between references, Title Case
    if escritura:
        # Replace ; with  |  separator for readability
        escritura_display = escritura.replace(";", "  |")
    else:
        escritura_display = ""

    # Animation Parameters — lasts 2 seconds (60 frames at 30fps)
    ANIM_DURATION = 60
    anim_progress = min(1.0, frame_num / ANIM_DURATION)
    ease_progress = 1.0 - math.pow(1.0 - anim_progress, 3)

    # Dimensions
    MARGIN_LEFT = 80
    BASE_Y = canvas_height - 220

    BADGE_W, BADGE_H = 200, 46
    TITLE_W, TITLE_H = 1600, 70
    SCRIPTURE_W, SCRIPTURE_H = 1600, 42

    # Horizontal sweep offset
    offset_x = -1500 * (1.0 - ease_progress)

    # 1. SCRIPTURE BAR (bottom layer)
    if escritura_display:
        s_x = MARGIN_LEFT + offset_x
        s_y = BASE_Y + TITLE_H
        scripture_bg = Image.new("RGBA", (SCRIPTURE_W, SCRIPTURE_H), SCRIPTURE_BG)
        frame.paste(scripture_bg, (int(s_x), int(s_y)), scripture_bg)

        s_font = get_font(22, "medium")
        draw.text((s_x + 20, s_y + 10), escritura_display, font=s_font, fill=(255, 255, 255, 255))

    # 2. TITLE BAR (middle layer — white background)
    t_x = MARGIN_LEFT + offset_x
    t_y = BASE_Y
    title_bg = Image.new("RGBA", (TITLE_W, TITLE_H), TITLE_BG)
    frame.paste(title_bg, (int(t_x), int(t_y)), title_bg)

    # Auto-scale font to fit title — try sizes from 40 down to 28
    display_title = title
    max_w = TITLE_W - 40

    for try_size in range(40, 27, -2):
        t_font = get_font(try_size, "bold")
        bbox = draw.textbbox((0, 0), display_title, font=t_font)
        if bbox[2] - bbox[0] <= max_w:
            break

    # If still too wide after scaling down, truncate as last resort
    bbox = draw.textbbox((0, 0), display_title, font=t_font)
    if bbox[2] - bbox[0] > max_w:
        while bbox[2] - bbox[0] > max_w - 40:
            display_title = display_title[:-1]
            bbox = draw.textbbox((0, 0), display_title, font=t_font)
        display_title += "..."

    # Subtle shadow + main text
    draw.text((t_x + 21, t_y + 14), display_title, font=t_font, fill=(0, 0, 0, 40))
    draw.text((t_x + 20, t_y + 13), display_title, font=t_font, fill=TEXT_COLOR)

    # 3. BADGE (top layer — "Ven Sígueme")
    l_x = MARGIN_LEFT + offset_x
    l_y = BASE_Y - BADGE_H
    badge_bg = Image.new("RGBA", (BADGE_W, BADGE_H), BADGE_BG)
    frame.paste(badge_bg, (int(l_x), int(l_y)), badge_bg)

    l_font = get_font(20, "semibold")
    badge_text = programa.upper()
    l_bbox = draw.textbbox((0, 0), badge_text, font=l_font)
    l_text_w = l_bbox[2] - l_bbox[0]
    l_text_h = l_bbox[3] - l_bbox[1]
    draw.text(
        (l_x + (BADGE_W - l_text_w) // 2, l_y + (BADGE_H - l_text_h) // 2 - 1),
        badge_text, font=l_font, fill=(20, 20, 20, 255)
    )


def draw_vertical_title(
    draw: ImageDraw.Draw,
    frame: Image.Image,
    title_highlighted: str,
    canvas_width: int,
    image_y: int,
    frame_num: int
) -> None:
    """
    Draw the title above the content image in vertical format with alternating colors.

    Words that are ALL UPPERCASE in title_highlighted render in CELESTE (sky blue).
    Other words render in WHITE. This creates the visual alternation seen in reels.

    The title is centered horizontally, positioned close above the image.
    Features a fade-in animation and word wrapping.
    """
    COLOR_WHITE = (255, 255, 255)
    COLOR_CELESTE = (100, 200, 240)  # Sky blue / celeste
    SHADOW_COLOR = (0, 0, 0)

    # Parse words with their color based on casing
    raw_words = title_highlighted.split()

    # Auto-scale font to fit width
    max_width = canvas_width - 120  # 60px margin each side
    font_size = 48

    # Find best font size by checking total text width
    plain_title = title_highlighted  # Use full text for sizing
    for try_size in range(48, 26, -2):
        font = get_font(try_size, "extrabold")
        bbox = draw.textbbox((0, 0), plain_title, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            font_size = try_size
            break
        font_size = try_size

    font = get_font(font_size, "extrabold")

    # Word wrapping — build lines of (word, is_highlight) tuples
    lines: list[list[tuple]] = []
    current_line_words = []
    current_line_text = ""

    for word in raw_words:
        test_text = f"{current_line_text} {word}".strip()
        bbox = draw.textbbox((0, 0), test_text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line_words.append(word)
            current_line_text = test_text
        else:
            if current_line_words:
                lines.append(current_line_words)
            current_line_words = [word]
            current_line_text = word
    if current_line_words:
        lines.append(current_line_words)

    # Calculate total text block height
    line_height = font_size + 12
    total_text_height = len(lines) * line_height

    # Position: centered, close above the image
    start_y = image_y - total_text_height - 25
    start_y = max(30, start_y)

    # Fade-in animation (first 30 frames = 1 second)
    if frame_num < 30:
        alpha = int(255 * (frame_num / 30.0))
    else:
        alpha = 255

    for i, line_words in enumerate(lines):
        # Calculate total line width for centering
        line_text = " ".join(line_words)
        bbox = draw.textbbox((0, 0), line_text, font=font)
        total_line_width = bbox[2] - bbox[0]
        start_x = (canvas_width - total_line_width) // 2
        y = start_y + i * line_height

        # Draw each word individually with the right color
        cursor_x = start_x
        for word_idx, word in enumerate(line_words):
            # Determine if this word is highlighted (ALL UPPERCASE, >1 char)
            is_highlight = word.isupper() and len(word) > 1
            color = COLOR_CELESTE if is_highlight else COLOR_WHITE

            # Stroke outline for readability
            stroke_alpha = int(alpha * 0.6)
            for dx in [-2, -1, 1, 2]:
                for dy in [-2, -1, 1, 2]:
                    draw.text((cursor_x + dx, y + dy), word, font=font, fill=(*SHADOW_COLOR, stroke_alpha))
            # Shadow
            draw.text((cursor_x + 2, y + 2), word, font=font, fill=(*SHADOW_COLOR, int(alpha * 0.5)))
            # Main text with color
            draw.text((cursor_x, y), word, font=font, fill=(*color, alpha))

            # Advance cursor
            word_bbox = draw.textbbox((0, 0), word + " ", font=font)
            cursor_x += word_bbox[2] - word_bbox[0]


def draw_vertical_bottom_tags(
    draw: ImageDraw.Draw,
    frame: Image.Image,
    programa: str,
    escritura_short: str,
    canvas_width: int,
    canvas_height: int,
    image_y: int,
    image_size: int,
    frame_num: int
) -> None:
    """
    Draw the bottom tags below the image in vertical format.
    Two rounded-rectangle pills side by side: "Ven Sígueme" and the scripture reference.
    Both use celeste/sky-blue background color matching the highlighted title words.
    """
    COLOR_CELESTE = (100, 200, 240)  # Same as title highlights
    TEXT_COLOR_DARK = (15, 25, 45)   # Dark navy text on celeste bg

    # Position: centered below the image
    tag_y = image_y + image_size + 20
    if tag_y > canvas_height - 70:
        tag_y = canvas_height - 70  # Don't go below screen

    # Fade-in animation (delayed, starts after title at frame 20)
    if frame_num < 20:
        alpha_f = 0.0
    elif frame_num < 50:
        alpha_f = (frame_num - 20) / 30.0
    else:
        alpha_f = 1.0

    if alpha_f <= 0:
        return

    tag_font = get_font(24, "bold")
    tag_height = 44
    padding_h = 24  # Horizontal padding inside each tag
    gap = 20        # Gap between the two tags
    corner_radius = 10

    # Calculate tag widths
    prog_text = programa
    esc_text = escritura_short

    prog_bbox = draw.textbbox((0, 0), prog_text, font=tag_font)
    prog_w = (prog_bbox[2] - prog_bbox[0]) + padding_h * 2

    esc_bbox = draw.textbbox((0, 0), esc_text, font=tag_font)
    esc_w = (esc_bbox[2] - esc_bbox[0]) + padding_h * 2

    total_width = prog_w + gap + esc_w
    start_x = (canvas_width - total_width) // 2

    alpha = int(255 * alpha_f)

    # Draw "Ven Sígueme" tag
    tag1_img = Image.new("RGBA", (prog_w, tag_height), (0, 0, 0, 0))
    tag1_draw = ImageDraw.Draw(tag1_img)
    tag1_draw.rounded_rectangle(
        [(0, 0), (prog_w - 1, tag_height - 1)],
        radius=corner_radius,
        fill=(*COLOR_CELESTE, alpha)
    )
    # Center text in tag
    prog_text_w = prog_bbox[2] - prog_bbox[0]
    prog_text_h = prog_bbox[3] - prog_bbox[1]
    prog_tx = (prog_w - prog_text_w) // 2
    prog_ty = (tag_height - prog_text_h) // 2 - 2
    tag1_draw.text((prog_tx, prog_ty), prog_text, font=tag_font, fill=(*TEXT_COLOR_DARK, alpha))
    frame.paste(tag1_img, (start_x, tag_y), tag1_img)

    # Draw scripture tag
    tag2_x = start_x + prog_w + gap
    tag2_img = Image.new("RGBA", (esc_w, tag_height), (0, 0, 0, 0))
    tag2_draw = ImageDraw.Draw(tag2_img)
    tag2_draw.rounded_rectangle(
        [(0, 0), (esc_w - 1, tag_height - 1)],
        radius=corner_radius,
        fill=(*COLOR_CELESTE, alpha)
    )
    esc_text_w = esc_bbox[2] - esc_bbox[0]
    esc_text_h = esc_bbox[3] - esc_bbox[1]
    esc_tx = (esc_w - esc_text_w) // 2
    esc_ty = (tag_height - esc_text_h) // 2 - 2
    tag2_draw.text((esc_tx, esc_ty), esc_text, font=tag_font, fill=(*TEXT_COLOR_DARK, alpha))
    frame.paste(tag2_img, (tag2_x, tag_y), tag2_img)


def create_frame(
    image: Image.Image,
    config: dict,
    title: str,
    video_format: str,
    canvas_width: int,
    canvas_height: int,
    background: Optional[Image.Image],
    image_progress: float = 0.5,
    frame_num: int = 0,
    total_frames: int = 1,
    title_highlighted: str = "",
    escritura_short: str = "",
) -> np.ndarray:
    """
    Create a single video frame with:
    1. Blurred decorative background
    2. Centered square content image with Ken Burns
    3. Lower third (horizontal) or title + bottom tags (vertical)
    """
    # Start with background
    if background is not None:
        frame = background.copy().convert("RGBA")
    else:
        frame = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 255))

    # Calculate image placement on canvas
    x, y, target_w, target_h = calculate_image_placement(
        canvas_width, canvas_height, video_format
    )

    # Apply Ken Burns to the content image, output at target size
    ken_burns_image = apply_ken_burns(image, image_progress, target_w, target_h)

    # Paste content image onto the background
    frame.paste(ken_burns_image.convert("RGB"), (x, y))

    draw = ImageDraw.Draw(frame)

    if video_format == "vertical":
        # Use highlighted title for celeste/white alternation, fallback to plain title
        display_title = title_highlighted if title_highlighted else title
        draw_vertical_title(draw, frame, display_title, canvas_width, y, frame_num)

        # Draw bottom tags: "Ven Sígueme" + scripture reference
        metadata = config.get("video_metadata", {})
        programa = metadata.get("programa", "Ven Sígueme")
        esc = escritura_short if escritura_short else metadata.get("escritura", "")
        if esc:
            draw_vertical_bottom_tags(
                draw, frame, programa, esc,
                canvas_width, canvas_height, y, target_w, frame_num
            )
    else:
        # Draw professional lower third at bottom left (existing)
        draw_professional_lower_third(
            draw, frame, title, frame_num, total_frames, config,
            canvas_width, canvas_height
        )

    return np.array(frame.convert("RGB"))


def get_current_image_index(
    frame_time: float,
    images: List[Tuple[str, Image.Image, float, float]],
    timestamps: dict
) -> Tuple[int, float]:
    """
    Determine which image should be shown at a given frame time.

    Uses start_time/end_time stored directly on each image tuple.
    The timestamps parameter is kept for signature compatibility but
    is no longer used when assets have direct time ranges.
    """
    if not images:
        return 0, 0.5

    for i, (asset_id, img, start_time, end_time) in enumerate(images):
        if start_time <= frame_time <= end_time:
            duration = end_time - start_time
            progress = (frame_time - start_time) / duration if duration > 0 else 0.5
            return i, max(0.0, min(1.0, progress))

    # Past all images -> show last one
    return len(images) - 1, 1.0


def save_metadata(script_data: dict, output_dir: Path) -> None:
    """Save video metadata as a full YouTube-ready text file."""
    metadata_path = output_dir / "metadata.txt"
    title_youtube = script_data.get("title_youtube", script_data.get("topic", ""))
    metadata = script_data.get("metadata", {})
    hashtags = metadata.get("hashtags", [])
    tags = metadata.get("tags", [])

    # Build description from new structured fields or legacy
    desc_hook = metadata.get("description_hook", "")
    desc_body = metadata.get("description_body", "")
    desc_bullets = metadata.get("description_bullets", [])
    escrituras = metadata.get("escrituras_mencionadas", [])
    legacy_desc = metadata.get("description", "")

    sep = "=" * 45
    line = "─" * 30

    lines = []
    lines.append(sep)
    lines.append("YOUTUBE TITLE:")
    lines.append(sep)
    lines.append(title_youtube)

    lines.append(sep)
    lines.append("DESCRIPTION:")
    lines.append(sep)

    if desc_hook:
        # New structured format
        lines.append(desc_hook)
        lines.append("")
        if desc_body:
            lines.append(desc_body)
            lines.append("")
        lines.append("En este video exploramos:")
        for bullet in desc_bullets:
            lines.append(bullet)
        lines.append("")
        lines.append(line)
        if escrituras:
            lines.append("📌 RECURSOS MENCIONADOS:")
            for esc in escrituras:
                lines.append(f"• {esc}")
            lines.append(line)
            lines.append("")
        lines.append("Si este mensaje te tocó el corazón, compártelo con alguien que necesite escucharlo. 💛")
        lines.append("")
        lines.append("🔔 SUSCRÍBETE a Relatos de Luz para más historias del Evangelio que fortalecen tu fe cada semana.")
        lines.append("💬 Deja tu comentario contándonos tu experiencia.")
        lines.append("👍 Dale LIKE si quieres más videos como este.")
        lines.append("")
        lines.append(line)
        lines.append("📱 Síguenos:")
        lines.append("Canal: @RelatosDeLuz")
        lines.append(line)
        lines.append("")
        lines.append(" ".join(hashtags))
    else:
        # Legacy single description field
        lines.append(legacy_desc)
        lines.append("")
        lines.append(" ".join(hashtags))

    lines.append("")
    lines.append(sep)
    lines.append("TAGS (copiar y pegar en YouTube Studio):")
    lines.append(sep)
    lines.append(",".join(tags))

    lines.append("")
    lines.append(sep)
    lines.append("HASHTAGS (ya incluidos en descripción):")
    lines.append(sep)
    lines.append(" ".join(hashtags))

    lines.append("")
    lines.append(sep)
    lines.append("NOTAS SEO:")
    lines.append(sep)
    if "|" in title_youtube:
        lines.append("- El título usa HOOK + contexto de búsqueda separado por '|'")
    lines.append("- Tags incluyen variaciones con y sin acentos para capturar más búsquedas")
    lines.append("- Las primeras 2 líneas de la descripción son visibles antes del 'mostrar más'")
    lines.append("- Emojis en descripción mejoran CTR")

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_video(output_path: Path, video_format: str = "horizontal") -> None:
    """
    Render the complete video with GPU acceleration.

    Args:
        output_path: Path for the output MP4
        video_format: "horizontal" (1920x1080) or "vertical" (1080x1920)
    """
    # Determine canvas dimensions
    if video_format == "vertical":
        canvas_width = VERTICAL_WIDTH    # 1080
        canvas_height = VERTICAL_HEIGHT  # 1920
    else:
        canvas_width = HORIZONTAL_WIDTH  # 1920
        canvas_height = HORIZONTAL_HEIGHT  # 1080

    format_label = f"{canvas_width}x{canvas_height} ({'9:16 Vertical' if video_format == 'vertical' else '16:9 Horizontal'})"

    update_render_progress("loading", 0, f"Loading configuration and assets ({format_label})...")
    print("Loading configuration and assets...")

    config = load_config()
    script = load_script()
    timestamps = load_timestamps()

    script_data = script.get("script", {})
    title = script_data.get("title_internal", script_data.get("topic", ""))
    title_highlighted = script_data.get("title_internal_highlighted", "")
    escritura_short = script_data.get("escritura_short", "")
    narration = script_data.get("narration", {})
    visual_assets = narration.get("visual_assets", [])

    images = load_images(visual_assets, timestamps.get("words", []))
    if not images:
        clear_render_progress()
        raise ValueError("No images found.")

    # Load and prepare the blurred decorative background
    # Pass visual_assets so it can fallback to a content image if bg.png is missing
    background = load_background_image(canvas_width, canvas_height, visual_assets)

    audio_path = AUDIO_DIR / "current.mp3"
    from moviepy.editor import AudioFileClip as TempAudioClip
    temp_audio = TempAudioClip(str(audio_path))
    audio_duration = temp_audio.duration
    temp_audio.close()

    LOOP_TAIL_SECONDS = 5.0
    video_duration = audio_duration + LOOP_TAIL_SECONDS
    total_frames = int(video_duration * FPS)

    print(f"Resolution: {format_label}")
    print(f"Duration: {video_duration:.1f}s | Frames: {total_frames} | FPS: {FPS}")
    print()

    update_render_progress("encoding", 0, f"0/{total_frames} frames")

    temp_video_path = output_path.with_suffix(".temp.mp4")
    container = av.open(str(temp_video_path), mode='w')

    try:
        stream = container.add_stream('hevc_nvenc', rate=FPS)
        stream.options = {'preset': 'p4', 'tune': 'hq', 'rc': 'vbr', 'cq': '23'}
        print("Encoder: NVIDIA NVENC (GPU)")
    except:
        stream = container.add_stream('libx264', rate=FPS)
        stream.options = {'preset': 'medium', 'crf': '23'}
        print("Encoder: libx264 (CPU)")

    stream.width = canvas_width
    stream.height = canvas_height
    stream.pix_fmt = 'yuv420p'

    render_start = _time_module.time()

    for frame_num in range(total_frames):
        frame_time = frame_num / FPS
        img_idx, img_progress = get_current_image_index(frame_time, images, timestamps)
        current_image = images[img_idx][1]

        frame_array = create_frame(
            current_image, config, title, video_format,
            canvas_width, canvas_height, background,
            img_progress, frame_num, total_frames,
            title_highlighted=title_highlighted,
            escritura_short=escritura_short,
        )
        video_frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
        video_frame = video_frame.reformat(format='yuv420p')

        for packet in stream.encode(video_frame):
            container.mux(packet)

        # Progress indicator every 30 frames (1 second of video)
        if frame_num % 30 == 0 or frame_num == total_frames - 1:
            pct = (frame_num + 1) / total_frames * 100
            elapsed = _time_module.time() - render_start
            fps_real = (frame_num + 1) / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_num - 1) / fps_real if fps_real > 0 else 0
            detail = f"Frame {frame_num+1}/{total_frames} | {fps_real:.1f} fps | ETA: {eta:.0f}s"
            print(f"\r  Rendering: {pct:5.1f}% | {detail}  ", end="", flush=True)
            update_render_progress("encoding", pct, detail)

    print()  # newline after progress

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # Merge audio
    update_render_progress("mixing_audio", 95, "Merging narration + background music...")
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, afx
    video_clip = VideoFileClip(str(temp_video_path))
    narration_clip = AudioFileClip(str(audio_path))

    music_filename = script_data.get("background_music", "Echoes_of_Starlight.mp3")
    music_path = MUSIC_DIR / music_filename
    if not music_path.exists(): music_path = MUSIC_DIR / "Echoes_of_Starlight.mp3"

    if music_path.exists():
        music_clip = AudioFileClip(str(music_path))
        if music_clip.duration < video_clip.duration:
            music_clip = afx.audio_loop(music_clip, nloops=int(video_clip.duration / music_clip.duration) + 1)
        music_clip = music_clip.subclip(0, video_clip.duration)

        narration_boosted = narration_clip.volumex(1.15)

        fade_start = video_duration - 3.0

        def music_volume(t):
            vol = np.where(
                t < audio_duration,
                0.15,
                np.where(
                    t < fade_start,
                    0.40,
                    0.40 * np.maximum(0, (video_duration - t) / 3.0)
                )
            )
            return vol

        def apply_music_volume(gf, t):
            audio_data = gf(t)
            vol = music_volume(t)
            if isinstance(vol, np.ndarray) and vol.ndim == 1 and audio_data.ndim == 2:
                vol = vol[:, np.newaxis]
            return audio_data * vol

        music_dynamic = music_clip.fl(apply_music_volume, keep_duration=True)

        final_audio = CompositeAudioClip([music_dynamic, narration_boosted]).set_duration(video_clip.duration)
    else:
        final_audio = narration_clip

    print("Merging audio...")
    final_clip = video_clip.set_audio(final_audio)
    update_render_progress("writing_final", 98, "Writing final MP4 with audio...")
    final_clip.write_videofile(str(output_path), codec='libx264', audio_codec='aac', logger=None)

    video_clip.close()
    narration_clip.close()
    if music_path.exists(): music_clip.close()
    temp_video_path.unlink(missing_ok=True)
    save_metadata(script_data, output_path.parent)

    update_render_progress("complete", 100, f"Render finished ({format_label})")
    clear_render_progress()

def main():
    import sys
    # Support command-line format argument: python video_renderer.py [horizontal|vertical]
    video_format = "horizontal"
    if len(sys.argv) > 1 and sys.argv[1] in ("horizontal", "vertical"):
        video_format = sys.argv[1]

    if video_format == "vertical":
        output_path = OUTPUT_DIR / "current_vertical.mp4"
    else:
        output_path = OUTPUT_DIR / "current_horizontal.mp4"

    render_video(output_path, video_format=video_format)

if __name__ == "__main__":
    main()
