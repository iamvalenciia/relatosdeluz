"""
Gemini Image Generator - Generate images using Gemini 2.0 Flash
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Installing google-genai...")
    import subprocess
    subprocess.check_call(["pip", "install", "google-genai", "pillow"])
    from google import genai
    from google.genai import types

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
SCRIPTS_DIR = DATA_DIR / "scripts"

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=GEMINI_API_KEY)


def load_script() -> dict:
    """Load the current script"""
    script_path = SCRIPTS_DIR / "current.json"
    with open(script_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_image(prompt: str, asset_id: str, retry_count: int = 3) -> bool:
    """
    Generate an image using Gemini 2.0 Flash.
    Images are generated in 1:1 square format.
    """
    formatted_prompt = (
        f"Generate an oil painting style image in 1:1 square aspect ratio: {prompt}. "
        f"Family friendly, spiritual, inspirational. High quality artwork. "
        f"The image MUST be in square 1:1 format, centered on the main subject."
    )

    for attempt in range(retry_count):
        try:
            print(f"  Generating {asset_id} (attempt {attempt + 1})...")

            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1"
                    )
                )
            )

            # Check for image in response
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        # Save the image
                        image_path = IMAGES_DIR / f"{asset_id}.jpeg"
                        with open(image_path, "wb") as f:
                            f.write(part.inline_data.data)
                        print(f"  Saved: {image_path}")
                        return True

            print(f"  No image in response")

        except Exception as e:
            error_msg = str(e)
            print(f"  Error: {error_msg[:100]}")

            if "503" in error_msg or "capacity" in error_msg.lower():
                if attempt < retry_count - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            elif attempt < retry_count - 1:
                time.sleep(5)

    return False


def main():
    """Main entry point"""
    print("=" * 50)
    print("GEMINI IMAGE GENERATOR (2.0 Flash)")
    print("=" * 50)

    # Ensure images directory exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load script
    script = load_script()
    visual_assets = script.get("script", {}).get("narration", {}).get("visual_assets", [])

    print(f"\nFound {len(visual_assets)} images to generate\n")

    success_count = 0

    for i, asset in enumerate(visual_assets, 1):
        asset_id = asset.get("visual_asset_id")
        prompt = asset.get("image_prompt")

        print(f"\n[{i}/{len(visual_assets)}] {asset_id}")
        print("-" * 40)

        # Check if image already exists
        if (IMAGES_DIR / f"{asset_id}.jpeg").exists():
            print(f"  Already exists, skipping...")
            success_count += 1
            continue

        if generate_image(prompt, asset_id):
            success_count += 1

        # Delay between generations to avoid rate limits
        if i < len(visual_assets):
            print("  Waiting 5s before next image...")
            time.sleep(5)

    print("\n" + "=" * 50)
    print(f"COMPLETE: {success_count}/{len(visual_assets)} images generated")
    print("=" * 50)


if __name__ == "__main__":
    main()
