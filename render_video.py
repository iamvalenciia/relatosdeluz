"""
Helamans Videos - Main Video Render Pipeline
Generates narration video from script and images.
"""

from pathlib import Path
from src.video_renderer import render_video

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "output"


def main():
    """Main entry point. Usage: python render_video.py [horizontal|vertical]"""
    import sys
    print("=" * 60)
    print("  HELAMANS VIDEOS - Ven Sígueme Video Generator")
    print("=" * 60)
    print()

    # Support command-line format argument
    video_format = "horizontal"
    if len(sys.argv) > 1 and sys.argv[1] in ("horizontal", "vertical"):
        video_format = sys.argv[1]

    if video_format == "vertical":
        output_path = OUTPUT_DIR / "current_vertical.mp4"
    else:
        output_path = OUTPUT_DIR / "current_horizontal.mp4"

    try:
        render_video(output_path, video_format=video_format)
        print()
        print(f"¡Video {video_format} generado exitosamente!")
        print(f"Ubicación: {output_path}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nAsegúrate de:")
        print("  1. Crear data/scripts/current.json con tu script")
        print("  2. Generar imágenes 1:1 en data/images/")
        print("  3. Ejecutar: python -m src.audio_generator")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        raise


if __name__ == "__main__":
    main()
