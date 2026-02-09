"""
Video Renderer - Hybrid rendering engine using PyAV + Pillow + NumPy
Uses NVIDIA GPU acceleration (hevc_nvenc) for RTX 3060 TI.

Features:
- Ken Burns effect (zoom + pan)
- Horizontal 1920x1080 (16:9) format for YouTube
- Professional TV news lower third overlay
- Opening sweep animation
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

# Video constants - HORIZONTAL 16:9 FORMAT
FPS = 30
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


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
    Images are resized to 1920x1080 (16:9) for horizontal video.
    
    Returns:
        List of (asset_id, PIL Image, start_word_index, end_word_index)
    """
    images = []
    for asset in visual_assets:
        asset_id = asset.get("visual_asset_id")
        img_path = find_image(asset_id)
        
        if img_path:
            img = Image.open(img_path).convert("RGB")
            # Resize to 1920x1080 (16:9) if needed
            if img.size != (VIDEO_WIDTH, VIDEO_HEIGHT):
                # Use cover resize to maintain aspect ratio
                img = resize_cover(img, VIDEO_WIDTH, VIDEO_HEIGHT)
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
        weight: 'bold', 'semibold', or 'medium'
    """
    fonts_dir = DATA_DIR / "fonts"

    weight_map = {
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
        image: Source image (1920x1080)
        progress: Animation progress (0.0 to 1.0)
        zoom_start: Initial zoom level
        zoom_end: Final zoom level
        pan_direction: Direction of pan ("center", "left", "right")

    Returns:
        Transformed image at 1920x1080
    """
    # Smooth easing function (ease-in-out cosine)
    ease_progress = 0.5 - 0.5 * math.cos(progress * math.pi)

    # Current zoom level (fully floating-point, no rounding)
    current_zoom = zoom_start + (zoom_end - zoom_start) * ease_progress

    target_w, target_h = VIDEO_WIDTH, VIDEO_HEIGHT
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

    # Build affine transform coefficients for 16:9
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
    config: dict
) -> None:
    """
    Draw a professional news-style lower third at the bottom left.
    Features an opening animation and smart text truncation.
    """
    # Colors (LDS Branding)
    CHURCH_BLUE = (0, 46, 93, 255)         # LDS Navy Blue
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
    BASE_Y = VIDEO_HEIGHT - 220

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
    t_font_size = 40

    for try_size in range(40, 27, -2):
        t_font = get_font(try_size, "bold")
        bbox = draw.textbbox((0, 0), display_title, font=t_font)
        if bbox[2] - bbox[0] <= max_w:
            t_font_size = try_size
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

    # (Sweep animation square removed — served no purpose)


def create_frame(
    image: Image.Image,
    config: dict,
    title: str,
    image_progress: float = 0.5,
    crossfade_alpha: float = 1.0,
    next_image: Optional[Image.Image] = None,
    frame_num: int = 0,
    total_frames: int = 1
) -> np.ndarray:
    """
    Create a single video frame with Ken Burns effect and professional lower third.
    """
    # Create base frame (pure black background)
    frame = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 255))
    
    # Apply Ken Burns to current image
    ken_burns_image = apply_ken_burns(image, image_progress)
    
    # Paste the image
    frame.paste(ken_burns_image.convert("RGB"), (0, 0))
    
    draw = ImageDraw.Draw(frame)
    
    # Draw professional lower third at bottom left
    draw_professional_lower_third(draw, frame, title, frame_num, total_frames, config)
    
    return np.array(frame.convert("RGB"))


def get_current_image_index(
    frame_time: float,
    images: List[Tuple[str, Image.Image, int, int]],
    timestamps: dict
) -> Tuple[int, float]:
    """Determine which image should be shown at a given time."""
    words = timestamps.get("words", [])
    if not images or not words:
        return 0, 0.5
    
    time_ranges = []
    for i, (asset_id, img, start_idx, end_idx) in enumerate(images):
        image_end_time = 0.0
        for word in words:
            if word.get("index", 0) == end_idx:
                image_end_time = word.get("end", 0.0)
                break
        time_ranges.append(image_end_time)
    
    prev_end = 0.0
    for i, end_time in enumerate(time_ranges):
        if prev_end <= frame_time <= end_time:
            duration = end_time - prev_end
            progress = (frame_time - prev_end) / duration if duration > 0 else 0.5
            return i, max(0.0, min(1.0, progress))
        prev_end = end_time
    
    return len(images) - 1, 1.0


def save_metadata(script_data: dict, output_dir: Path) -> None:
    """Save video metadata to a text file."""
    metadata_path = output_dir / "metadata.txt"
    title_youtube = script_data.get("title_youtube", script_data.get("topic", ""))
    metadata = script_data.get("metadata", {})
    description = metadata.get("description", "")
    hashtags = metadata.get("hashtags", [])
    
    content = f"TITLE: {title_youtube}\n\nDESCRIPTION:\n{description}\n\n{' '.join(hashtags)}"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(content)


def render_video(output_path: Path) -> None:
    """Render the complete video with GPU acceleration."""
    print("Loading configuration and assets...")
    
    config = load_config()
    script = load_script()
    timestamps = load_timestamps()
    
    script_data = script.get("script", {})
    title = script_data.get("title_internal", script_data.get("topic", ""))
    narration = script_data.get("narration", {})
    visual_assets = narration.get("visual_assets", [])
    
    images = load_images(visual_assets)
    if not images:
        raise ValueError("No images found.")
    
    audio_path = AUDIO_DIR / "current.mp3"
    from moviepy.editor import AudioFileClip as TempAudioClip
    temp_audio = TempAudioClip(str(audio_path))
    audio_duration = temp_audio.duration
    temp_audio.close()
    
    LOOP_TAIL_SECONDS = 5.0
    video_duration = audio_duration + LOOP_TAIL_SECONDS
    total_frames = int(video_duration * FPS)

    print(f"Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT} (16:9 Horizontal)")
    print(f"Duration: {video_duration:.1f}s | Frames: {total_frames} | FPS: {FPS}")
    print()

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

    stream.width = VIDEO_WIDTH
    stream.height = VIDEO_HEIGHT
    stream.pix_fmt = 'yuv420p'

    import time as _time
    render_start = _time.time()

    for frame_num in range(total_frames):
        frame_time = frame_num / FPS
        img_idx, img_progress = get_current_image_index(frame_time, images, timestamps)
        current_image = images[img_idx][1]

        frame_array = create_frame(current_image, config, title, img_progress, 1.0, None, frame_num, total_frames)
        video_frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
        video_frame = video_frame.reformat(format='yuv420p')

        for packet in stream.encode(video_frame):
            container.mux(packet)

        # Progress indicator every 30 frames (1 second of video)
        if frame_num % 30 == 0 or frame_num == total_frames - 1:
            pct = (frame_num + 1) / total_frames * 100
            elapsed = _time.time() - render_start
            fps_real = (frame_num + 1) / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_num - 1) / fps_real if fps_real > 0 else 0
            print(f"\r  Rendering: {pct:5.1f}% | Frame {frame_num+1}/{total_frames} | {fps_real:.1f} fps | ETA: {eta:.0f}s  ", end="", flush=True)

    print()  # newline after progress
    
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    
    # Merge audio
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

        # Volume calibration:
        # - During narration: music at 15% (subtle background)
        # - After narration ends: music rises to 40% then fades out
        # - Narration boosted to 1.15x for clarity
        narration_boosted = narration_clip.volumex(1.15)

        # Music: low during voice, rises after voice ends, fade out last 3 seconds
        # Uses np.where because moviepy passes t as numpy arrays
        fade_start = video_duration - 3.0

        def music_volume(t):
            vol = np.where(
                t < audio_duration,
                0.15,                                         # During narration
                np.where(
                    t < fade_start,
                    0.40,                                     # After narration, before fade
                    0.40 * np.maximum(0, (video_duration - t) / 3.0)  # Fade out
                )
            )
            return vol

        music_dynamic = music_clip.fl(lambda gf, t: gf(t) * music_volume(t), keep_duration=True)

        final_audio = CompositeAudioClip([music_dynamic, narration_boosted]).set_duration(video_clip.duration)
    else:
        final_audio = narration_clip

    print("Merging audio...")
    final_clip = video_clip.set_audio(final_audio)
    final_clip.write_videofile(str(output_path), codec='libx264', audio_codec='aac', logger=None)

    video_clip.close()
    narration_clip.close()
    if music_path.exists(): music_clip.close()
    temp_video_path.unlink(missing_ok=True)
    save_metadata(script_data, output_path.parent)

def main():
    output_path = OUTPUT_DIR / "current.mp4"
    render_video(output_path)

if __name__ == "__main__":
    main()
