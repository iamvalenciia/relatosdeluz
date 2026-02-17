
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

def generate_image_1a():
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = (
        "A detailed oil painting in the style of Latter-day Saint sacred art (similar to Walter Rane, Arnold Friberg), "
        "depicting a dramatic cinematic scene of a young Abraham bound on a stone sacrificial altar, "
        "his father standing behind him with a conflicted expression. In the foreground, Abraham's face shows "
        "determination and quiet courage, his eyes looking upward. Behind them, a dark pagan temple with flames "
        "and idol statues looms in the background. Multiple depth layers: Abraham in close foreground, father in midground, "
        "dark temple in background. High contrast between warm firelight on Abraham's face and the dark oppressive atmosphere. "
        "Rich warm color palette with deep reds and golds against shadows. Anatomically correct proportions, "
        "naturalistic facial features, dignified expressions. Space left on upper left for text overlay. "
        "Latter-day Saint sacred art style. Square 1:1 composition, family friendly. "
        "NO CROSSES, NO HALOS, NO WINGS on angels, NO Catholic imagery."
    )
    
    # Format the prompt as the project does
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
        f"If angels appear: normal human figures in white robes, NO WINGS, no halos. "
        f"STRICTLY FORBIDDEN: NO crosses, NO crucifixes, NO halos, NO rosaries, "
        f"NO Catholic imagery, NO Orthodox icons, NO stained glass windows, "
        f"NO baby cherubs with wings, NO crown of thorns, NO graphic suffering, "
        f"NO surrealism, NO bizarre elements, NO exaggerated emotions, "
        f"NO glowing eyes, NO golden tears, NO fantasy elements, "
        f"NO text, NO letters, NO numbers visible in the image. "
        f"Family friendly, Latter-day Saint reverent atmosphere. "
        f"Square 1:1 composition. Ultra-detailed, high resolution artwork."
    )

    print("Generating 1a.png in 1:1...")
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
                image_path = IMAGES_DIR / "1a.png"
                # Backup existing
                if image_path.exists():
                    os.replace(image_path, IMAGES_DIR / "1a_backup.png")
                image.save(str(image_path))
                print(f"SUCCESS: Saved 1a.png (1:1)")
                return True
        print("FAILED: No image in response")
    except Exception as e:
        print(f"ERROR: {e}")
    return False

if __name__ == "__main__":
    generate_image_1a()
