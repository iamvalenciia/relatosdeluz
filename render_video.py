"""
Helamans Videos - Main Video Render Pipeline
Generates narration video from script and images.
"""

from pathlib import Path
from src.video_renderer import render_video

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "output"


def main():
    """Main entry point."""
    print("=" * 60)
    print("  HELAMANS VIDEOS - Ven Sígueme Video Generator")
    print("=" * 60)
    print()
    
    output_path = OUTPUT_DIR / "current.mp4"
    
    try:
        render_video(output_path)
        print()
        print("¡Video generado exitosamente!")
        print(f"Ubicación: {output_path}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nAsegúrate de:")
        print("  1. Crear data/scripts/current.json con tu script")
        print("  2. Agregar imágenes 1080x1080 a data/images/")
        print("  3. Ejecutar: python -m src.audio_generator")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        raise


if __name__ == "__main__":
    main()
