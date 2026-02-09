"""
Video Renderer - Hybrid rendering engine using PyAV + Pillow + NumPy
Uses NVIDIA GPU acceleration (hevc_nvenc) for RTX 3060 TI.

Features:
- Ken Burns effect (zoom + pan)
- Centered 1080x1080 images in 1080x1920 frame
- TV news lower third overlay with program and scripture
- Title overlay at top
- Image sync with audio timestamps
"""

import os
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

# Video constants
FPS = 30
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920


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


def load_images(visual_assets: List[dict]) -> List[Tuple[str, Image.Image, int, int]]:
    """
    Load all images referenced in visual_assets.
    
    Returns:
        List of (asset_id, PIL Image, start_word_index, end_word_index)
    """
    images = []
    for asset in visual_assets:
        asset_id = asset.get("visual_asset_id")
        img_path = find_image(asset_id)
        
        if img_path:
            img = Image.open(img_path).convert("RGBA")
            # Resize to 1080x1080 if needed
            if img.size != (1080, 1080):
                img = img.resize((1080, 1080), Image.Resampling.LANCZOS)
            images.append((
                asset_id,
                img,
                asset.get("start_word_index", 0),
                asset.get("end_word_index", 0)
            ))
            print(f"Loaded image: {asset_id} ({img_path.name})")
        else:
            print(f"WARNING: Image not found for asset_id: {asset_id}")
    
    return images


def get_title_font(size: int) -> ImageFont.FreeTypeFont:
    """Get font for title - EBGaramond Bold"""
    fonts_dir = DATA_DIR / "fonts"
    
    title_fonts = [
        fonts_dir / "EBGaramond-Bold.ttf",
        fonts_dir / "PlayfairDisplay-Bold.ttf",
        "C:/Windows/Fonts/georgiab.ttf",
    ]
    
    for font_path in title_fonts:
        path = str(font_path) if hasattr(font_path, 'exists') else font_path
        try:
            if (hasattr(font_path, 'exists') and font_path.exists()) or \
               (isinstance(font_path, str) and os.path.exists(font_path)):
                return ImageFont.truetype(path, size)
        except:
            continue
    
    return ImageFont.load_default()


def get_footer_font(size: int) -> ImageFont.FreeTypeFont:
    """Get font for footer - EBGaramond Bold (same as title)"""
    fonts_dir = DATA_DIR / "fonts"
    
    footer_fonts = [
        fonts_dir / "EBGaramond-Bold.ttf",
        fonts_dir / "Cinzel-Bold.ttf",
        "C:/Windows/Fonts/georgia.ttf",
    ]
    
    for font_path in footer_fonts:
        path = str(font_path) if hasattr(font_path, 'exists') else font_path
        try:
            if (hasattr(font_path, 'exists') and font_path.exists()) or \
               (isinstance(font_path, str) and os.path.exists(font_path)):
                return ImageFont.truetype(path, size)
        except:
            continue
    
    return ImageFont.load_default()


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def truncate_for_lower_third(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.Draw
) -> str:
    """Truncate text with '...' if it exceeds max_width in pixels."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    if text_width <= max_width:
        return text

    ellipsis = "..."
    ellipsis_bbox = draw.textbbox((0, 0), ellipsis, font=font)
    ellipsis_width = ellipsis_bbox[2] - ellipsis_bbox[0]
    available_width = max_width - ellipsis_width

    truncated = text
    while len(truncated) > 1:
        truncated = truncated[:-1]
        bbox = draw.textbbox((0, 0), truncated, font=font)
        if bbox[2] - bbox[0] <= available_width:
            break

    # Try to break at a word boundary
    last_space = truncated.rfind(' ')
    if last_space > len(truncated) * 0.6:
        truncated = truncated[:last_space]

    return truncated.rstrip() + ellipsis


def draw_lower_third(
    draw: ImageDraw.Draw,
    config: dict,
    title: str,
    programa: str,
    escritura: str,
    image_y: int,
    image_height: int = 1080
) -> None:
    """
    Draw a TV news-style lower third overlay on the frame.

    Features a two-tier design:
    - Top bar (blue): Video title, truncated with '...' if needed
    - Bottom bar (orange): Program name + scripture reference
    - White accent line between bars

    Positioned in the bottom-left of the frame, overlaying the image.
    """
    style = config.get("style", {})
    lt_config = style.get("lower_third", {})

    if not lt_config.get("enabled", True):
        return

    # --- Dimensions ---
    width_percent = lt_config.get("width_percent", 70)
    lt_width = int(VIDEO_WIDTH * width_percent / 100)
    lt_left = lt_config.get("left_margin", 30)
    lt_right = lt_left + lt_width
    bottom_margin = lt_config.get("bottom_margin", 20)

    # Top bar config
    top_cfg = lt_config.get("top_bar", {})
    top_height = top_cfg.get("height", 60)
    top_color = hex_to_rgb(top_cfg.get("color", "#005DA6"))
    top_font_size = top_cfg.get("font_size", 32)
    top_text_color = hex_to_rgb(top_cfg.get("text_color", "#FFFFFF"))
    top_padding_h = top_cfg.get("padding_h", 20)

    # Accent config
    accent_cfg = lt_config.get("accent_line", {})
    accent_height = accent_cfg.get("height", 4)
    accent_color = hex_to_rgb(accent_cfg.get("color", "#FFFFFF"))

    # Bottom bar config
    bot_cfg = lt_config.get("bottom_bar", {})
    bot_height = bot_cfg.get("height", 44)
    bot_color = hex_to_rgb(bot_cfg.get("color", "#E87722"))
    bot_font_size = bot_cfg.get("font_size", 24)
    bot_text_color = hex_to_rgb(bot_cfg.get("text_color", "#FFFFFF"))
    bot_padding_h = bot_cfg.get("padding_h", 20)

    # --- Vertical position ---
    image_bottom = image_y + image_height
    total_height = top_height + accent_height + bot_height
    lt_bottom = image_bottom - bottom_margin
    lt_top = lt_bottom - total_height

    top_bar_y = lt_top
    top_bar_bottom = top_bar_y + top_height
    accent_y = top_bar_bottom
    accent_bottom = accent_y + accent_height
    bot_bar_y = accent_bottom
    bot_bar_bottom = bot_bar_y + bot_height

    # --- Fonts ---
    top_font = get_footer_font(top_font_size)
    bot_font = get_footer_font(bot_font_size)

    # --- Draw top bar (blue) ---
    draw.rectangle(
        [(lt_left, top_bar_y), (lt_right, top_bar_bottom)],
        fill=top_color
    )

    # --- Draw accent line ---
    draw.rectangle(
        [(lt_left, accent_y), (lt_right, accent_bottom)],
        fill=accent_color
    )

    # --- Draw bottom bar (orange) ---
    draw.rectangle(
        [(lt_left, bot_bar_y), (lt_right, bot_bar_bottom)],
        fill=bot_color
    )

    # --- Title text (top bar) with truncation ---
    available_text_width = lt_width - (2 * top_padding_h)
    display_title = truncate_for_lower_third(title, top_font, available_text_width, draw)

    # Vertically center in the top bar
    ref_bbox = draw.textbbox((0, 0), "Mg", font=top_font)
    text_height = ref_bbox[3] - ref_bbox[1]
    text_y = top_bar_y + (top_height - text_height) // 2
    text_x = lt_left + top_padding_h

    draw.text((text_x, text_y), display_title, font=top_font, fill=top_text_color)

    # --- Bottom bar text ---
    programa_upper = programa.upper()

    # Calculate space division (~55% programa, ~45% escritura)
    usable_width = lt_width - (2 * bot_padding_h)
    programa_zone_width = int(usable_width * 0.55)
    divider_x = lt_left + bot_padding_h + programa_zone_width

    # Draw vertical divider on bottom bar
    if escritura:
        divider_padding = 8
        draw.line(
            [(divider_x, bot_bar_y + divider_padding),
             (divider_x, bot_bar_bottom - divider_padding)],
            fill=bot_text_color,
            width=1
        )

    # Draw programa text (left-aligned in its zone)
    bot_ref_bbox = draw.textbbox((0, 0), "Mg", font=bot_font)
    bot_text_height = bot_ref_bbox[3] - bot_ref_bbox[1]
    bot_text_y = bot_bar_y + (bot_height - bot_text_height) // 2

    draw.text(
        (lt_left + bot_padding_h, bot_text_y),
        programa_upper,
        font=bot_font,
        fill=bot_text_color
    )

    # Draw escritura text (after divider)
    if escritura:
        escritura_x = divider_x + 12
        draw.text(
            (escritura_x, bot_text_y),
            escritura,
            font=bot_font,
            fill=bot_text_color
        )


def apply_ken_burns(
    image: Image.Image,
    progress: float,
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
        image: Source image (1080x1080)
        progress: Animation progress (0.0 to 1.0)
        zoom_start: Initial zoom level
        zoom_end: Final zoom level
        pan_direction: Direction of pan ("center", "left", "right")

    Returns:
        Transformed image at 1080x1080
    """
    # Smooth easing function (ease-in-out cosine)
    ease_progress = 0.5 - 0.5 * math.cos(progress * math.pi)

    # Current zoom level (fully floating-point, no rounding)
    current_zoom = zoom_start + (zoom_end - zoom_start) * ease_progress

    target_size = 1080
    img_w, img_h = image.size

    # The visible region in source-image coordinates:
    # At zoom=1.0 we see the full image; at zoom=1.08 we see a smaller crop (zoomed in).
    # crop_size = image_size / zoom  (how many source pixels map to the output)
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

    # Build affine transform coefficients:
    # PIL AFFINE maps output (x,y) -> source (x',y') as:
    #   x' = a*x + b*y + c
    #   y' = d*x + e*y + f
    # We want: source_x = cx + x * (crop_w / target_size)
    #          source_y = cy + y * (crop_h / target_size)
    scale_x = crop_w / target_size
    scale_y = crop_h / target_size

    affine_coeffs = (
        scale_x, 0.0, cx,
        0.0, scale_y, cy,
    )

    # BICUBIC resampling gives smooth sub-pixel interpolation with no jitter
    result = image.transform(
        (target_size, target_size),
        Image.AFFINE,
        affine_coeffs,
        resample=Image.Resampling.BICUBIC,
    )

    return result


def create_frame(
    image: Image.Image,
    config: dict,
    title: str,
    image_progress: float = 0.5,
    crossfade_alpha: float = 1.0,
    next_image: Optional[Image.Image] = None
) -> np.ndarray:
    """
    Create a single video frame with Ken Burns effect, title, and lower third overlay.
    
    Args:
        image: Current image to display
        config: Configuration dictionary
        title: Video title
        image_progress: Progress through current image (0-1)
        crossfade_alpha: Alpha for crossfade (1.0 = fully current image)
        next_image: Optional next image for crossfade
    
    Returns:
        NumPy array of the frame (RGB)
    """
    # Create base frame (pure black background)
    frame = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0))
    
    # Apply Ken Burns to current image
    ken_burns_image = apply_ken_burns(image, image_progress)
    
    # Handle crossfade if needed
    if next_image is not None and crossfade_alpha < 1.0:
        next_ken_burns = apply_ken_burns(next_image, 0.0)
        # Blend images
        ken_burns_image = Image.blend(
            next_ken_burns.convert("RGB"),
            ken_burns_image.convert("RGB"),
            crossfade_alpha
        )
    
    # Center image in frame (vertical center with space for title and footer)
    image_y = (VIDEO_HEIGHT - 1080) // 2
    frame.paste(ken_burns_image.convert("RGB"), (0, image_y))
    
    draw = ImageDraw.Draw(frame)
    
    # Get style config
    style = config.get("style", {})
    title_font_size = style.get("font_size_title", 62)
    
    # Calculate title position - positioned close to the top edge of the image
    title_padding = 50  # Padding from screen edges
    title_margin_from_image = 50  # Margin between title bottom and image top
    max_title_width = VIDEO_WIDTH - (title_padding * 2)

    # Function to wrap text into lines that fit within max_width
    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
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

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    # Use original font size, allow up to 3 lines
    title_font = get_title_font(title_font_size)
    lines = wrap_text(title, title_font, max_title_width)

    # Get line height
    ref_bbox = draw.textbbox((0, 0), "Mg", font=title_font)
    line_height = ref_bbox[3] - ref_bbox[1]
    line_spacing = 10

    total_height = line_height * len(lines) + line_spacing * (len(lines) - 1)
    start_y = image_y - total_height - title_margin_from_image

    # Get highlight color from config (gold for emphasized words in CAPS)
    highlight_color = hex_to_rgb(style.get("title_highlight_color", "#FFD700"))
    shadow_color_hex = style.get("title_shadow_color", "#000000")
    shadow_color = hex_to_rgb(shadow_color_hex) + (200,)

    # Draw each line centered with word-level highlighting
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_width = bbox[2] - bbox[0]
        line_x = (VIDEO_WIDTH - line_width) // 2
        y = start_y + i * (line_height + line_spacing)

        # Draw word by word to support highlight colors
        # Words that are fully UPPERCASE (2+ chars) get highlighted in gold
        words = line.split(' ')
        current_x = line_x
        for wi, word in enumerate(words):
            # Determine if this word should be highlighted
            is_highlighted = len(word) >= 2 and word.replace('¿', '').replace('?', '').replace('¡', '').replace('!', '').replace(',', '').replace('.', '').isupper()
            word_color = highlight_color if is_highlighted else (255, 255, 255)

            display_word = word + (' ' if wi < len(words) - 1 else '')

            # Shadow (stronger for readability)
            draw.text((current_x + 2, y + 2), display_word, font=title_font, fill=shadow_color)
            draw.text((current_x + 1, y + 1), display_word, font=title_font, fill=shadow_color)
            # Main text
            draw.text((current_x, y), display_word, font=title_font, fill=word_color)

            word_bbox = draw.textbbox((0, 0), display_word, font=title_font)
            current_x += word_bbox[2] - word_bbox[0]
    
    # Draw TV news-style lower third overlay
    metadata = config.get("video_metadata", {})
    programa = metadata.get("programa", "Ven Sígueme")
    escritura = metadata.get("escritura", "")

    draw_lower_third(draw, config, title, programa, escritura, image_y)
    
    return np.array(frame)


def get_current_image_index(
    frame_time: float,
    images: List[Tuple[str, Image.Image, int, int]],
    timestamps: dict
) -> Tuple[int, float]:
    """
    Determine which image should be shown at a given time.
    Uses TIME-BASED interpolation for smooth Ken Burns effect.
    Pre-computes CONTINUOUS time ranges to eliminate gaps.
    
    Returns:
        (image_index, progress_within_image) where progress is 0.0 to 1.0
    """
    words = timestamps.get("words", [])
    
    if not images or not words:
        return 0, 0.5
    
    # Pre-compute time ranges for all images
    # Use CONTINUOUS ranges (end of image N = start of image N+1)
    time_ranges = []
    
    for i, (asset_id, img, start_idx, end_idx) in enumerate(images):
        # Find the end time of the last word in this image
        image_end_time = 0.0
        for word in words:
            if word.get("index", 0) == end_idx:
                image_end_time = word.get("end", 0.0)
                break
        time_ranges.append(image_end_time)
    
    # Now determine which image based on continuous ranges
    # Image 0: 0 to time_ranges[0]
    # Image 1: time_ranges[0] to time_ranges[1]
    # etc.
    
    prev_end = 0.0
    for i, end_time in enumerate(time_ranges):
        if prev_end <= frame_time <= end_time:
            # Calculate smooth progress within this image
            duration = end_time - prev_end
            if duration > 0:
                progress = (frame_time - prev_end) / duration
            else:
                progress = 0.5
            return i, max(0.0, min(1.0, progress))
        prev_end = end_time
    
    # Default to last image at full progress
    return len(images) - 1, 1.0


def render_video(output_path: Path) -> None:
    """
    Render the complete video with GPU acceleration.
    
    Uses NVIDIA NVENC (hevc_nvenc) for hardware encoding on RTX 3060 TI.
    """
    print("Loading configuration and assets...")
    
    config = load_config()
    script = load_script()
    timestamps = load_timestamps()
    
    # Get script data
    script_data = script.get("script", {})
    title = script_data.get("topic", "")
    narration = script_data.get("narration", {})
    visual_assets = narration.get("visual_assets", [])
    
    # Load images
    images = load_images(visual_assets)
    
    if not images:
        raise ValueError("No images found. Please add images to data/images/")
    
    # Get audio duration from actual audio file (more reliable than timestamps)
    audio_path = AUDIO_DIR / "current.mp3"
    if not audio_path.exists():
        raise FileNotFoundError("Audio not found. Run audio_generator.py first.")
    
    # Use MoviePy to get accurate duration
    from moviepy.editor import AudioFileClip as TempAudioClip
    temp_audio = TempAudioClip(str(audio_path))
    audio_duration = temp_audio.duration
    temp_audio.close()
    
    # Add 2.5 seconds of music-only tail for smooth loop transition
    # The narration ends, music continues alone for ~2.5s, then video restarts
    LOOP_TAIL_SECONDS = 2.5
    video_duration = audio_duration + LOOP_TAIL_SECONDS
    total_frames = int(video_duration * FPS)

    print(f"Video duration: {audio_duration:.2f}s narration + {LOOP_TAIL_SECONDS}s loop tail = {video_duration:.2f}s ({total_frames} frames)")
    print(f"Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
    print(f"Title: {title}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Temporary video path (without audio)
    temp_video_path = output_path.with_suffix(".temp.mp4")
    
    # Setup PyAV output with NVIDIA GPU encoding
    print("\nInitializing video encoder (NVIDIA NVENC)...")
    
    container = av.open(str(temp_video_path), mode='w')
    
    # Try NVIDIA NVENC first, fall back to libx264 if not available
    try:
        stream = container.add_stream('hevc_nvenc', rate=FPS)
        stream.options = {
            'preset': 'p4',  # Quality preset (p1-p7, p4 is balanced)
            'tune': 'hq',    # High quality tuning
            'rc': 'vbr',     # Variable bitrate
            'cq': '23',      # Quality level (lower = better)
        }
        print("Using NVIDIA NVENC (hevc_nvenc) for GPU acceleration")
    except av.error.InvalidDataError:
        print("NVIDIA NVENC not available, falling back to libx264")
        stream = container.add_stream('libx264', rate=FPS)
        stream.options = {'preset': 'medium', 'crf': '23'}
    
    stream.width = VIDEO_WIDTH
    stream.height = VIDEO_HEIGHT
    stream.pix_fmt = 'yuv420p'
    
    # Render frames
    print("\nRendering frames...")
    
    for frame_num in range(total_frames):
        frame_time = frame_num / FPS
        
        # Get current image and progress
        img_idx, img_progress = get_current_image_index(frame_time, images, timestamps)
        
        # Get current and next images for potential crossfade
        current_image = images[img_idx][1]
        next_image = images[img_idx + 1][1] if img_idx + 1 < len(images) else None
        
        # Calculate crossfade alpha (fade in last 10% of image duration)
        crossfade_alpha = 1.0
        if img_progress > 0.9 and next_image:
            crossfade_alpha = 1.0 - (img_progress - 0.9) / 0.1
        
        # Create frame
        frame_array = create_frame(
            current_image,
            config,
            title,
            img_progress,
            crossfade_alpha,
            next_image
        )
        
        # Convert to PyAV frame
        video_frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
        video_frame = video_frame.reformat(format='yuv420p')
        
        # Encode frame
        for packet in stream.encode(video_frame):
            container.mux(packet)
        
        # Progress indicator
        if frame_num % (FPS * 5) == 0:  # Every 5 seconds
            percent = (frame_num / total_frames) * 100
            print(f"  Progress: {percent:.1f}% ({frame_time:.1f}s / {video_duration:.1f}s)")
    
    # Flush remaining frames
    for packet in stream.encode():
        container.mux(packet)
    
    container.close()
    
    print("\nMerging audio with background music...")
    
    # Merge audio with video using moviepy
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, afx
    
    video_clip = VideoFileClip(str(temp_video_path))
    # No fade out - video ends clean for loop effect
    narration_clip = AudioFileClip(str(audio_path))
    
    # Get duration from validation clip
    video_full_duration = video_clip.duration
    
    # Look for background music from script or use default
    music_filename = script_data.get("background_music", "Echoes_of_Starlight.mp3")
    music_path = MUSIC_DIR / music_filename

    # Fallback to default if specified music not found
    if not music_path.exists():
        print(f"Music '{music_filename}' not found, trying default...")
        music_path = MUSIC_DIR / "Echoes_of_Starlight.mp3"
    
    if music_path.exists():
        print(f"Adding background music: {music_path.name}")
        music_clip = AudioFileClip(str(music_path))
        
        # Loop music if shorter than video duration
        if music_clip.duration < video_full_duration:
            # Calculate how many loops needed
            loops_needed = int(video_full_duration / music_clip.duration) + 1
            music_clip = afx.audio_loop(music_clip, nloops=loops_needed)
        
        # Trim music to match video duration exactly (narration + 3s buffer)
        music_clip = music_clip.subclip(0, video_full_duration)
        
        # Apply fade-out to music at the end (0.5 seconds) - REMOVED per user request
        # music_clip = music_clip.audio_fadeout(0.5)
        
        # Reduce music volume (20% of original) so narration is clear
        music_clip = music_clip.volumex(0.20)
        
        # Composite audio: narration on top of music
        # IMPORTANT: Set explicit duration to prevent audio looping/bleeding
        final_audio = CompositeAudioClip([music_clip, narration_clip]).set_duration(video_full_duration)
        
        # Apply fade-out to final audio - REMOVED per user request to keep music volume up
        # final_audio = final_audio.audio_fadeout(0.3)
    else:
        print("No background music found, using narration only")
        # No fade out - clean ending for loop
        final_audio = narration_clip
    
    # Set combined audio
    final_clip = video_clip.set_audio(final_audio)
    
    # Export final video with proper audio settings
    final_clip.write_videofile(
        str(output_path),
        codec='libx264',
        audio_codec='aac',
        audio_bitrate='192k',  # Higher quality audio
        fps=FPS,
        preset='medium',
        temp_audiofile=str(output_path.with_suffix('.temp.aac')),
        remove_temp=True,
        write_logfile=False,
        verbose=False,
        logger=None
    )
    
    # Cleanup
    video_clip.close()
    narration_clip.close()
    if music_path.exists():
        music_clip.close()
    temp_video_path.unlink(missing_ok=True)
    
    print(f"\n{'=' * 50}")
    print(f"Video saved to: {output_path}")
    print(f"{'=' * 50}")


def main():
    """Main entry point for video rendering."""
    print("=" * 50)
    print("HELAMANS VIDEOS - Video Renderer")
    print("NVIDIA GPU Acceleration Enabled")
    print("=" * 50)
    
    output_path = OUTPUT_DIR / "current.mp4"
    render_video(output_path)


if __name__ == "__main__":
    main()
