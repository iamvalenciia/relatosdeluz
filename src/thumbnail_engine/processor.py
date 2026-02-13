"""
Image Processor - Background removal and visual effects.

Uses rembg for offline background removal and Pillow for all visual effects.
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

from .workspace import WORKSPACE_DIR, add_asset, resolve_asset_path, get_manifest, save_manifest


def remove_bg(asset_id: str) -> dict:
    """
    Remove background from an image using rembg (offline, no API).
    Saves result as {original_name}_nobg.png in workspace.
    """
    try:
        from rembg import remove
    except ImportError:
        raise ImportError(
            "rembg not installed. Run: pip install rembg\n"
            "First run will download the U2-Net model (~176MB)."
        )

    src_path = resolve_asset_path(asset_id)
    print(f"  Removing background from: {src_path.name}")

    with open(src_path, "rb") as f:
        input_data = f.read()

    output_data = remove(input_data)

    stem = src_path.stem
    nobg_id = f"{stem}_nobg"
    output_path = WORKSPACE_DIR / f"{nobg_id}.png"
    with open(output_path, "wb") as f:
        f.write(output_data)

    print(f"  Saved: {output_path.name}")
    info = add_asset(nobg_id, "element_nobg", output_path, f"Background removed from {asset_id}")
    return {"asset_id": nobg_id, "path": str(output_path), **info}


def apply_effects(target: str, effects: list) -> dict:
    """
    Apply a pipeline of visual effects to an image.

    Each effect: {"type": "vignette", "params": {"strength": 0.4}}
    Supported: vignette, blur, color_grade, glow, border,
               brightness, contrast, saturation, gradient
    """
    if target in ("composed", "composed.png"):
        img_path = WORKSPACE_DIR / "composed.png"
    else:
        img_path = resolve_asset_path(target)

    if not img_path.exists():
        raise FileNotFoundError(f"Target not found: {img_path}")

    img = Image.open(img_path).convert("RGBA")
    print(f"  Applying {len(effects)} effect(s) to {img_path.name}")

    for effect in effects:
        etype = effect.get("type", "")
        params = effect.get("params", {})

        if etype == "vignette":
            img = _vignette(img, params)
        elif etype == "blur":
            img = _blur(img, params)
        elif etype == "color_grade":
            img = _color_grade(img, params)
        elif etype == "glow":
            img = _glow(img, params)
        elif etype == "border":
            img = _border(img, params)
        elif etype == "brightness":
            factor = params.get("factor", 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)
        elif etype == "contrast":
            factor = params.get("factor", 1.3)
            img = ImageEnhance.Contrast(img).enhance(factor)
        elif etype == "saturation":
            factor = params.get("factor", 1.2)
            img = ImageEnhance.Color(img).enhance(factor)
        elif etype == "gradient":
            img = _gradient_overlay(img, params)
        else:
            print(f"  WARNING: Unknown effect '{etype}', skipping")

        print(f"    Applied: {etype}")

    # Save back
    out_path = WORKSPACE_DIR / "composed.png"
    img.convert("RGB").save(str(out_path), "PNG", quality=95)
    print(f"  Saved: {out_path.name}")
    return {"path": str(out_path), "effects_applied": len(effects)}


# ── Effect Implementations ────────────────────────────────────────────

def _vignette(img: Image.Image, params: dict) -> Image.Image:
    """Darken edges for cinematic look."""
    strength = params.get("strength", 0.4)
    color = tuple(params.get("color", [0, 0, 0]))
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cx, cy = w / 2, h / 2
    max_dist = (cx ** 2 + cy ** 2) ** 0.5

    arr = np.zeros((h, w, 4), dtype=np.uint8)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    normalized = np.clip(dist / max_dist, 0, 1)
    alpha = (normalized ** 1.5 * strength * 255).astype(np.uint8)

    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    arr[:, :, 3] = alpha
    overlay = Image.fromarray(arr, "RGBA")
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def _blur(img: Image.Image, params: dict) -> Image.Image:
    """Gaussian blur."""
    radius = params.get("radius", 5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _color_grade(img: Image.Image, params: dict) -> Image.Image:
    """Apply color grading presets."""
    preset = params.get("preset", "warm")
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)

    if preset == "warm":
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.08, 0, 255)  # +red
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.92, 0, 255)  # -blue
    elif preset == "cool":
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.92, 0, 255)  # -red
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.08, 0, 255)  # +blue
    elif preset == "dramatic":
        arr = np.clip(arr * 1.15 - 15, 0, 255)  # boost contrast
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.85, 0, 255)  # -blue
    elif preset == "golden_hour":
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.12, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.82, 0, 255)
    elif preset == "desaturated":
        gray = np.mean(arr, axis=2, keepdims=True)
        arr = arr * 0.4 + gray * 0.6

    result = Image.fromarray(arr.astype(np.uint8), "RGB")
    if img.mode == "RGBA":
        result = result.convert("RGBA")
        result.putalpha(img.getchannel("A"))
    return result


def _glow(img: Image.Image, params: dict) -> Image.Image:
    """Add glow effect by blending blurred bright version."""
    radius = params.get("radius", 20)
    intensity = params.get("intensity", 0.4)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    enhancer = ImageEnhance.Brightness(blurred)
    bright = enhancer.enhance(1.5)
    return Image.blend(img, bright, alpha=intensity)


def _border(img: Image.Image, params: dict) -> Image.Image:
    """Add colored border."""
    width = params.get("width", 4)
    color = params.get("color", "#FFD700")
    if isinstance(color, str):
        color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    bordered = img.copy()
    draw = ImageDraw.Draw(bordered)
    w, h = img.size
    for i in range(width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=(*color, 255))
    return bordered


def _gradient_overlay(img: Image.Image, params: dict) -> Image.Image:
    """Add directional gradient overlay."""
    direction = params.get("direction", "bottom_to_top")
    color = tuple(params.get("color", [0, 0, 0]))
    start_opacity = params.get("start_opacity", 0)
    end_opacity = params.get("end_opacity", 200)
    start_pct = params.get("start_percent", 50)

    w, h = img.size
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if direction == "bottom_to_top":
        start_y = int(h * start_pct / 100)
        span = h - start_y
        if span > 0:
            for y in range(start_y, h):
                progress = (y - start_y) / span
                alpha = int(start_opacity + (end_opacity - start_opacity) * progress)
                draw.rectangle([0, y, w, y + 1], fill=(*color, min(alpha, 255)))
    elif direction == "left_to_right":
        end_x = int(w * (100 - start_pct) / 100)
        if end_x > 0:
            for x in range(0, end_x):
                progress = 1.0 - (x / end_x)
                alpha = int(start_opacity + (end_opacity - start_opacity) * progress)
                draw.rectangle([x, 0, x + 1, h], fill=(*color, min(alpha, 255)))
    elif direction == "top_to_bottom":
        end_y = int(h * (100 - start_pct) / 100)
        if end_y > 0:
            for y in range(0, end_y):
                progress = 1.0 - (y / end_y)
                alpha = int(start_opacity + (end_opacity - start_opacity) * progress)
                draw.rectangle([0, y, w, y + 1], fill=(*color, min(alpha, 255)))

    return Image.alpha_composite(rgba, overlay)
