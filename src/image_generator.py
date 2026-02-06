"""
Image Generator - Sequential image generation with Gemini 2.5 Flash Image.
Generates images for video visual assets using Nano Banana model.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
SCRIPTS_DIR = DATA_DIR / "scripts"

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def load_script() -> dict:
    """Load the current script from data/scripts/current.json"""
    script_path = SCRIPTS_DIR / "current.json"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found at {script_path}. "
            "Please create your script JSON first."
        )
    with open(script_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_image_prompt(original_prompt: str) -> str:
    """Format the image prompt for Gemini with detailed oil painting instructions."""
    return (
        f"Create a museum-quality oil painting with the following scene: "
        f"{original_prompt}. "
        f"CRITICAL STYLE REQUIREMENTS: "
        f"Render this as a traditional oil painting with visible brushstrokes, "
        f"rich color depth, warm golden tones, and dramatic chiaroscuro lighting "
        f"in the style of the Old Masters (Rembrandt, Caravaggio). "
        f"All human figures must have anatomically correct proportions, "
        f"naturalistic facial features with dignified expressions, "
        f"and realistic skin tones. "
        f"NO surrealism, NO bizarre elements, NO exaggerated emotions, "
        f"NO glowing eyes, NO golden tears, NO fantasy elements. "
        f"NO text, NO letters, NO numbers visible in the image. "
        f"Family friendly, reverent spiritual atmosphere. "
        f"Square 1:1 aspect ratio. Ultra-detailed, high resolution artwork."
    )


def get_visual_assets(script: dict) -> list:
    """Extract visual assets from script."""
    narration = script.get("script", {}).get("narration", {})
    return narration.get("visual_assets", [])


def generate_image_with_gemini(prompt: str, asset_id: str, retry_count: int = 3) -> bool:
    """
    Generate an image using Gemini 2.5 Flash Image (Nano Banana).

    Args:
        prompt: The formatted image prompt
        asset_id: The asset ID for file naming
        retry_count: Number of retries on failure

    Returns:
        True if successful, False otherwise
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  ERROR: google-genai not installed. Run: pip install google-genai")
        return False

    if not GEMINI_API_KEY:
        print("  ERROR: GEMINI_API_KEY not found in .env")
        return False

    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(retry_count):
        try:
            print(f"  Generating (attempt {attempt + 1}/{retry_count})...")

            # Use Gemini 2.5 Flash Image (Nano Banana) for image generation
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                    )
                )
            )

            # Extract and save the image from response parts
            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    image_path = IMAGES_DIR / f"{asset_id}.png"
                    image.save(str(image_path))
                    print(f"  SUCCESS: Saved {image_path.name}")
                    return True

            print(f"  WARNING: No image in response")

        except Exception as e:
            error_msg = str(e)
            print(f"  ERROR: {error_msg[:150]}")

            # Check for capacity/rate limit errors
            if "503" in error_msg or "capacity" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt < retry_count - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            elif attempt < retry_count - 1:
                time.sleep(5)

    return False


def generate_all_images(force: bool = False) -> tuple:
    """
    Generate all missing images.

    Args:
        force: If True, regenerate all images even if they exist

    Returns:
        Tuple of (success_count, total_count)
    """
    script = load_script()
    visual_assets = get_visual_assets(script)

    success_count = 0
    total_count = len(visual_assets)

    for i, asset in enumerate(visual_assets, 1):
        asset_id = asset.get("visual_asset_id")
        original_prompt = asset.get("image_prompt", "")
        formatted_prompt = format_image_prompt(original_prompt)

        print(f"\n[{i}/{total_count}] {asset_id}")
        print("-" * 40)

        # Check if image already exists
        image_exists = False
        for ext in [".jpeg", ".jpg", ".png", ".webp"]:
            if (IMAGES_DIR / f"{asset_id}{ext}").exists():
                image_exists = True
                break

        if image_exists and not force:
            print(f"  SKIP: Already exists")
            success_count += 1
            continue

        if generate_image_with_gemini(formatted_prompt, asset_id):
            success_count += 1

        # Delay between generations
        if i < total_count:
            time.sleep(3)

    return success_count, total_count


def check_images_exist() -> tuple:
    """Check which images exist and which are missing."""
    script = load_script()
    visual_assets = get_visual_assets(script)

    existing = []
    missing = []

    for asset in visual_assets:
        asset_id = asset.get("visual_asset_id")
        found = False
        for ext in [".jpeg", ".jpg", ".png", ".webp"]:
            if (IMAGES_DIR / f"{asset_id}{ext}").exists():
                existing.append(asset_id)
                found = True
                break
        if not found:
            missing.append(asset_id)

    return existing, missing


def print_generation_instructions():
    """Print instructions for manual image generation (fallback)."""
    print("=" * 60)
    print("MANUAL IMAGE GENERATION INSTRUCTIONS")
    print("=" * 60)
    print("\nIf automatic generation fails, generate images manually:")
    print("Format: 1:1 (square, 1080x1080)")
    print("Style: Oil painting")
    print("\n" + "-" * 60)

    script = load_script()
    visual_assets = get_visual_assets(script)

    for i, asset in enumerate(visual_assets, 1):
        asset_id = asset.get("visual_asset_id")
        prompt = format_image_prompt(asset.get("image_prompt", ""))
        print(f"\n[Image {i}] Asset ID: {asset_id}")
        print("-" * 40)
        print(prompt)
        print()

    print("=" * 60)
    print(f"Save images to: {IMAGES_DIR}")
    print("Naming: 1a.jpeg, 1b.jpeg, 1c.jpeg, etc.")
    print("=" * 60)


def main():
    """Main entry point for image generation."""
    print("=" * 50)
    print("HELAMANS VIDEOS - Image Generator")
    print("=" * 50)

    # Ensure images directory exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        existing, missing = check_images_exist()

        if existing:
            print(f"\nExisting images: {', '.join(existing)}")

        if missing:
            print(f"\nMissing images: {', '.join(missing)}")
            print("\n" + "-" * 50)
            print("Starting automatic generation with Gemini 2.5 Flash Image...")

            success, total = generate_all_images()

            print("\n" + "=" * 50)
            print(f"RESULT: {success}/{total} images generated")
            print("=" * 50)

            if success < total:
                print("\nSome images failed. Manual generation instructions:")
                print_generation_instructions()
        else:
            print("\nAll images present!")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please generate the script first.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nFallback to manual generation:")
        print_generation_instructions()


if __name__ == "__main__":
    main()
