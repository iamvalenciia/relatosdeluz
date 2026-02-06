"""
Video Renderer - Hybrid rendering engine using PyAV + Pillow + NumPy
Uses NVIDIA GPU acceleration (hevc_nvenc) for RTX 3060 TI.

Features:
- Ken Burns effect (zoom + pan)
- Centered 1080x1080 images in 1080x1920 frame
- Blue footer with program, date, scripture
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


def apply_ken_burns(
    image: Image.Image,
    progress: float,
    zoom_start: float = 1.0,
    zoom_end: float = 1.08,  # Subtle zoom for smooth effect
    pan_direction: str = "center"
) -> Image.Image:
    """
    Apply Ken Burns effect using affine transform for sub-pixel smooth animation.
    No integer rounding = no jitter.
    
    Args:
        image: Source image (1080x1080)
        progress: Animation progress (0.0 to 1.0)
        zoom_start: Initial zoom level
        zoom_end: Final zoom level
        pan_direction: Direction of pan ("center", "left", "right")
    
    Returns:
        Transformed image at 1080x1080
    """
    # Smooth easing function (ease-in-out)
    ease_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
    
    # Calculate current zoom
    current_zoom = zoom_start + (zoom_end - zoom_start) * ease_progress
    
    # Target size after scaling
    target_size = 1080
    
    # Scale the image up by the zoom factor
    scaled_size = int(target_size * current_zoom)
    
    # Scale image using high-quality resampling
    scaled_image = image.resize((scaled_size, scaled_size), Image.Resampling.LANCZOS)
    
    # Calculate crop position to center
    offset = (scaled_size - target_size) / 2.0
    
    # Apply panning offset
    if pan_direction == "left":
        pan_offset = offset * (1.0 - ease_progress)
    elif pan_direction == "right":
        pan_offset = offset * ease_progress
    else:
        pan_offset = 0
    
    # Calculate crop box (use floats, PIL handles sub-pixel with good interpolation)
    left = offset + pan_offset
    top = offset
    right = left + target_size
    bottom = top + target_size
    
    # Crop to final size - PIL handles sub-pixel cropping smoothly
    result = scaled_image.crop((int(left), int(top), int(right), int(bottom)))
    
    # Ensure exact size
    if result.size != (target_size, target_size):
        result = result.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
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
    Create a single video frame with Ken Burns effect, title, and footer.
    
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
    footer_color = hex_to_rgb(style.get("footer_color", "#1A3A5C"))
    title_font_size = style.get("font_size_title", 62)
    footer_font_size = style.get("font_size_footer", 22)
    
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

    # Draw each line centered
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_width = bbox[2] - bbox[0]
        x = (VIDEO_WIDTH - line_width) // 2
        y = start_y + i * (line_height + line_spacing)

        # Shadow
        draw.text((x + 2, y + 2), line, font=title_font, fill=(0, 0, 0, 180))
        # Text
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255))
    
    # Draw header bar (now a footer below the image)
    header_height = 80  # Increased height for better spacing
    header_y = image_y + 1080  # Position immediately below the image

    # Header background
    draw.rectangle(
        [(0, header_y), (VIDEO_WIDTH, header_y + header_height)],
        fill=footer_color
    )

    # Single divider in the center (2 sections: programa | escritura)
    divider_x = VIDEO_WIDTH // 2
    draw.line(
        [(divider_x, header_y + 10), (divider_x, header_y + header_height - 10)],
        fill=(255, 255, 255, 100),
        width=1
    )

    # Get header metadata (only programa and escritura, no fecha)
    metadata = config.get("video_metadata", {})
    programa = metadata.get("programa", "Ven Sígueme")
    escritura = metadata.get("escritura", "")

    footer_font = get_footer_font(footer_font_size)

    # Draw header text - centered in each section with text wrapping
    section_width = VIDEO_WIDTH // 2
    section_padding = 30  # Padding from dividers
    max_text_width = section_width - (section_padding * 2)

    def draw_wrapped_text(text, center_x, available_width, bar_y, bar_height):
        """Draw text, wrapping to two lines if needed"""
        # Get consistent line height using a reference character
        ref_bbox = draw.textbbox((0, 0), "Mg", font=footer_font)  # Use chars with ascender/descender
        line_height = ref_bbox[3] - ref_bbox[1]

        bbox = draw.textbbox((0, 0), text, font=footer_font)
        text_width = bbox[2] - bbox[0]

        # Ajuste visual para centrar mejor el texto (compensar descenders)
        vertical_offset = -6

        if text_width <= available_width:
            # Text fits on one line - center vertically using consistent height
            x = center_x - text_width // 2
            y = bar_y + (bar_height - line_height) // 2 + vertical_offset
            draw.text((x, y), text, font=footer_font, fill=(255, 255, 255))
        else:
            # Split text into two lines
            words = text.split()
            mid = len(words) // 2
            line1 = ' '.join(words[:mid]) if mid > 0 else words[0]
            line2 = ' '.join(words[mid:]) if mid > 0 else ' '.join(words[1:])

            # Calculate positions for two lines
            bbox1 = draw.textbbox((0, 0), line1, font=footer_font)
            bbox2 = draw.textbbox((0, 0), line2, font=footer_font)
            line1_width = bbox1[2] - bbox1[0]
            line2_width = bbox2[2] - bbox2[0]

            line_spacing = 3
            total_height = line_height * 2 + line_spacing
            text_start_y = bar_y + (bar_height - total_height) // 2 + vertical_offset

            x1 = center_x - line1_width // 2
            x2 = center_x - line2_width // 2

            draw.text((x1, text_start_y), line1, font=footer_font, fill=(255, 255, 255))
            draw.text((x2, text_start_y + line_height + line_spacing), line2, font=footer_font, fill=(255, 255, 255))

    # Program (left section) - center of first half
    section1_center = section_width // 2
    draw_wrapped_text(programa, section1_center, max_text_width, header_y, header_height)

    # Scripture (right section) - center of second half
    section2_center = divider_x + section_width // 2
    draw_wrapped_text(escritura, section2_center, max_text_width, header_y, header_height)
    
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
    
    # Minimal buffer for clean loop ending (video ends shortly after narration)
    video_duration = audio_duration + 0.5  # 0.5 seconds for clean cut, no long fade
    total_frames = int(video_duration * FPS)
    
    print(f"Video duration: {audio_duration:.2f} seconds + 3s buffer = {video_duration:.2f}s ({total_frames} frames)")
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
