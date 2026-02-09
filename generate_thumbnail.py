#!/usr/bin/env python3
"""
Generate a YouTube thumbnail for the current Ven Sígueme video.

Usage:
    python generate_thumbnail.py
    python generate_thumbnail.py --image data/images/1a.png
    python generate_thumbnail.py --title "Custom Title"
    python generate_thumbnail.py --output data/output/my_thumb.png

Configuration: data/thumbnail_config.json
Fonts needed: Download from Google Fonts → data/fonts/
  - Montserrat-ExtraBold.ttf
  - Montserrat-Bold.ttf
"""

import argparse
from pathlib import Path
from src.thumbnail_generator import generate_thumbnail


def main():
    parser = argparse.ArgumentParser(
        description="Generate YouTube thumbnail for Ven Sígueme video"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to source image (default: auto-select from video images)"
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default=None,
        help="Override title text (default: from script)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (default: data/output/thumbnail.png)"
    )

    args = parser.parse_args()

    image_path = Path(args.image) if args.image else None
    output_path = Path(args.output) if args.output else None

    generate_thumbnail(
        output_path=output_path,
        title_override=args.title,
        image_path_override=image_path
    )


if __name__ == "__main__":
    main()
