"""Regenerate image 1g with stronger no-wings angel restriction."""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_DIR = Path.cwd()
IMAGES_DIR = BASE_DIR / "data" / "images"


def generate_image_1g():
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Original prompt from current.json for 1g, with even stronger no-wings language
    prompt = (
        "A detailed oil painting in the style of Latter-day Saint sacred art "
        "(similar to Walter Rane, Simon Dewey), depicting a powerful heavenly messenger "
        "descending from heaven to rescue Abraham from the sacrificial altar. "
        "Abraham lies bound on the stone altar, looking up with awe and relief. "
        "The angel is a dignified man in flowing white robes, he has NO WINGS, "
        "he is a normal human being with arms visible at his sides, reaching down with "
        "authority to stop the sacrifice. He does NOT have wings of any kind. "
        "His back is clear of any wings or feathers. He looks like a glorified human man. "
        "Brilliant golden-white divine light radiates from above, pushing back the darkness "
        "of the pagan temple. The idol priests recoil in the background. "
        "Rich warm color palette with dramatic contrast between divine light and temple darkness. "
        "Visible brushstrokes, powerful composition. Anatomically correct proportions, "
        "naturalistic facial features, expressions of divine power and human relief. "
        "Latter-day Saint sacred art style. Square 1:1 composition, family friendly. "
        "NO CROSSES, NO HALOS, NO WINGS on angels, NO Catholic imagery."
    )

    formatted_prompt = (
        f"Create a museum-quality oil painting in the style of Latter-day Saint "
        f"sacred art (similar to artists Greg Olsen, Del Parson, Walter Rane, "
        f"Simon Dewey) with the following scene: {prompt}. "
        f"CRITICAL STYLE REQUIREMENTS: "
        f"Traditional oil painting with visible brushstrokes, rich warm color "
        f"palette, golden lighting, and dramatic chiaroscuro. "
        f"All human figures must have anatomically correct proportions, "
        f"naturalistic facial features with dignified expressions, "
        f"and realistic skin tones. "
        f"RELIGIOUS STYLE: Exclusively Latter-day Saint (LDS/Mormon) visual tradition. "
        f"If Jesus Christ appears: brown hair to shoulders, short neat beard, "
        f"white or cream robe, red or dark blue mantle, compassionate and strong expression. "
        f"ANGELS MUST NEVER HAVE WINGS. In Latter-day Saint doctrine, angels are "
        f"glorified human beings without wings. If angels appear: depict them as "
        f"normal dignified men in flowing white robes, standing or walking on the ground, "
        f"absolutely NO WINGS of any kind, no feathered wings, no ethereal wings, no halos. "
        f"STRICTLY FORBIDDEN: NO crosses, NO crucifixes, NO halos, NO rosaries, "
        f"NO Catholic imagery, NO Orthodox icons, NO stained glass windows, "
        f"NO baby cherubs with wings, NO crown of thorns, NO graphic suffering, "
        f"NO surrealism, NO bizarre elements, NO exaggerated emotions, "
        f"NO glowing eyes, NO golden tears, NO fantasy elements, "
        f"NO text, NO letters, NO numbers visible in the image. "
        f"Family friendly, Latter-day Saint reverent atmosphere. "
        f"Square 1:1 composition. Ultra-detailed, high resolution artwork."
    )

    for attempt in range(3):
        print(f"\nAttempt {attempt + 1}/3 - Generating 1g.png...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[formatted_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                    )
                )
            )

            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    image_path = IMAGES_DIR / "1g.png"
                    # Backup existing
                    if image_path.exists():
                        backup_path = IMAGES_DIR / "1g_backup.png"
                        os.replace(image_path, backup_path)
                        print(f"  Backed up existing 1g.png -> 1g_backup.png")
                    image.save(str(image_path))
                    print(f"SUCCESS: Saved 1g.png (1:1)")
                    return True
            print("No image in response, retrying...")
        except Exception as e:
            print(f"ERROR: {e}")
            if attempt < 2:
                print("Waiting 30s before retry...")
                time.sleep(30)
    
    print("FAILED after 3 attempts")
    return False


if __name__ == "__main__":
    generate_image_1g()
