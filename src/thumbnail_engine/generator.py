"""
Image Generator - Gemini 2.5 Flash Image API wrapper for thumbnail assets.

Generates three types of assets:
  - Backgrounds (16:9): dramatic scenes, environments, atmospheres
  - Elements (1:1): characters, objects, symbols on simple backgrounds
  - Text art (1:1): stylized text renderings
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from .workspace import WORKSPACE_DIR, next_id, add_asset

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Style Prefixes ────────────────────────────────────────────────────

BG_STYLES = {
    "cinematic": (
        "Ultra-dramatic cinematic scene with deep contrast, volumetric lighting, "
        "rich saturated colors, film-quality composition. "
    ),
    "painterly": (
        "Museum-quality oil painting with visible brushstrokes, "
        "rich warm color palette, golden lighting, dramatic chiaroscuro. "
    ),
    "dramatic": (
        "High-contrast dramatic scene with intense chiaroscuro lighting, "
        "deep shadows, bold colors, almost theatrical atmosphere. "
    ),
    "ethereal": (
        "Ethereal, dreamlike atmosphere with soft golden light, "
        "gentle color transitions, luminous sky, sacred feeling. "
    ),
    "photorealistic": (
        "Photorealistic high-resolution scene with natural lighting, "
        "accurate colors, sharp details, professional photography look. "
    ),
    "dark_abstract": (
        "Very DARK abstract background with minimal detail. Deep blacks, "
        "subtle color gradients, wisps of smoke or mist. NO recognizable objects "
        "or scenery. Pure atmospheric mood. Dark blue-black or dark red-black tones. "
        "Suitable as a backdrop that does NOT compete with a foreground subject. "
    ),
    "moody_bokeh": (
        "Extremely dark background with soft out-of-focus bokeh light points. "
        "Deep shadows, very shallow depth of field effect. Warm or cool color cast. "
        "The background should be 80%+ dark/black with subtle light accents. "
        "NO recognizable objects. Pure atmosphere and mood. "
    ),
    "dark_gradient": (
        "Solid dark gradient background transitioning from one dark color to another. "
        "For example: dark navy to black, dark crimson to black, dark forest to black. "
        "Minimal texture, clean and simple. NO objects, NO scenery, NO details. "
        "Pure color gradient for thumbnail composition. "
    ),
}

ELEMENT_STYLES = {
    "painterly": (
        "Oil painting style, visible brushstrokes, rich warm colors, "
        "dignified naturalistic features, LDS sacred art tradition. "
    ),
    "realistic": (
        "Highly realistic digital art, detailed features, natural skin tones, "
        "professional lighting, sharp focus. "
    ),
    "sacred_art": (
        "Latter-day Saint sacred art style (Greg Olsen, Del Parson, Walter Rane). "
        "Oil painting, golden lighting, reverent atmosphere. "
    ),
    "close_up_portrait": (
        "Dramatic CLOSE-UP portrait from chest up, face fills most of the frame. "
        "Cinematic portrait photography framing. Strong directional lighting "
        "(Rembrandt or split lighting) creating dramatic shadows on the face. "
        "Rich oil painting style in the LDS sacred art tradition "
        "(Greg Olsen, Walter Rane). The subject's expression must be INTENSE "
        "and clearly readable - eyes are the focal point. "
        "Detailed facial features, realistic skin texture, strong emotion. "
    ),
}

TEXT_STYLES = {
    "bold_gold": "Bold 3D golden metallic text with dramatic lighting and reflections",
    "fire": "Text made of flames and fire, dramatic burning effect",
    "glowing": "Glowing luminous text with ethereal light rays emanating outward",
    "ancient_carved": "Text carved into ancient stone, weathered and monumental",
    "neon": "Bright neon glowing text against dark background",
    "ice": "Frozen crystalline ice text with frost and cold blue tones",
    "stone": "Heavy stone text with realistic texture, cracks, and moss",
    "blood": "Dark red dripping text, dramatic horror-style lettering",
}

THUMBNAIL_SUFFIX = (
    " IMPORTANT: This image is for a YouTube thumbnail background. "
    "Make it visually DARK and MOODY with deep shadows. "
    "HIGH CONTRAST with minimal detail so a foreground subject can stand out. "
    "NO text, NO letters, NO numbers, NO watermarks in the image. "
    "NO people or faces. Clean atmospheric background only."
)

ELEMENT_BG_SUFFIX = (
    " The subject must be on a PLAIN SOLID WHITE BACKGROUND with no other "
    "elements, objects, or scenery. Just the subject isolated on pure white. "
    "This is for cutout/extraction purposes. "
    "FRAMING: Show the subject from CHEST UP (bust/portrait framing). "
    "The FACE must be clearly visible and take up a large portion of the image. "
    "Strong facial expression. Eyes must be the focal point."
)


def _call_gemini(prompt: str, aspect_ratio: str = "16:9", retry_count: int = 3) -> Path | None:
    """Call Gemini 2.5 Flash Image and save result. Returns path or None."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(retry_count):
        try:
            print(f"  Generating (attempt {attempt + 1}/{retry_count})...")
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                ),
            )
            for part in response.parts:
                if part.inline_data is not None:
                    return part.as_image()
            print("  WARNING: No image in response")
        except Exception as e:
            error_msg = str(e)
            print(f"  ERROR: {error_msg[:150]}")
            if "503" in error_msg or "capacity" in error_msg.lower():
                if attempt < retry_count - 1:
                    wait = 30 * (attempt + 1)
                    print(f"  Waiting {wait}s before retry...")
                    time.sleep(wait)
            elif attempt < retry_count - 1:
                time.sleep(5)

    return None


def generate_background(prompt: str, style: str = "cinematic") -> dict:
    """Generate a 16:9 background image for thumbnails."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    style_prefix = BG_STYLES.get(style, BG_STYLES["cinematic"])
    full_prompt = style_prefix + prompt + THUMBNAIL_SUFFIX

    print(f"  Style: {style}")
    print(f"  Prompt: {prompt[:100]}...")

    image = _call_gemini(full_prompt, aspect_ratio="16:9")
    if image is None:
        raise RuntimeError("Failed to generate background image after retries")

    asset_id = next_id("bg_")
    path = WORKSPACE_DIR / f"{asset_id}.png"
    image.save(str(path))
    print(f"  Saved: {path.name}")

    info = add_asset(asset_id, "background", path, prompt)
    return {"asset_id": asset_id, "path": str(path), **info}


def generate_element(prompt: str, style: str = "painterly") -> dict:
    """Generate a 1:1 element image (character, object) on white background."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    style_prefix = ELEMENT_STYLES.get(style, ELEMENT_STYLES["painterly"])
    full_prompt = style_prefix + prompt + ELEMENT_BG_SUFFIX

    print(f"  Style: {style}")
    print(f"  Prompt: {prompt[:100]}...")

    image = _call_gemini(full_prompt, aspect_ratio="1:1")
    if image is None:
        raise RuntimeError("Failed to generate element image after retries")

    asset_id = next_id("el_")
    path = WORKSPACE_DIR / f"{asset_id}.png"
    image.save(str(path))
    print(f"  Saved: {path.name}")

    info = add_asset(asset_id, "element", path, prompt)
    return {"asset_id": asset_id, "path": str(path), **info}


def generate_text_art(text: str, style: str = "bold_gold") -> dict:
    """Generate stylized text as an image."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    style_desc = TEXT_STYLES.get(style, TEXT_STYLES["bold_gold"])
    full_prompt = (
        f'Create an image of the text "{text}" rendered as: {style_desc}. '
        f"The text should be the ONLY element in the image, centered, "
        f"on a solid black or very dark background. "
        f"Make it dramatic and visually stunning. High resolution."
    )

    print(f"  Text: {text}")
    print(f"  Style: {style}")

    image = _call_gemini(full_prompt, aspect_ratio="1:1")
    if image is None:
        raise RuntimeError("Failed to generate text art after retries")

    asset_id = next_id("text_")
    path = WORKSPACE_DIR / f"{asset_id}.png"
    image.save(str(path))
    print(f"  Saved: {path.name}")

    info = add_asset(asset_id, "text_art", path, f"{text} [{style}]")
    return {"asset_id": asset_id, "path": str(path), **info}
