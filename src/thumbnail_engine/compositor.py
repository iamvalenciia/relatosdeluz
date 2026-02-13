"""
Compositor - Layer-based image composition engine with text overlay.

Composes multiple image layers into a single 1280x720 thumbnail with
precise control over positioning, scaling, rotation, opacity, and anchoring.
"""

from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw, ImageFont

from .workspace import (
    WORKSPACE_DIR, resolve_asset_path, get_manifest, save_manifest,
)

# Thumbnail dimensions
THUMB_W = 1280
THUMB_H = 720

# Fonts directory
FONTS_DIR = Path(__file__).parent.parent.parent / "data" / "fonts"


def get_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font with fallback chain."""
    # Try project fonts
    path = FONTS_DIR / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    # Try Windows system fonts
    import sys
    if sys.platform == "win32":
        winpath = Path("C:/Windows/Fonts") / name
        if winpath.exists():
            return ImageFont.truetype(str(winpath), size)
    # Fallback
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        print(f"  WARNING: Font '{name}' not found, using default")
        return ImageFont.load_default()


def hex_to_rgba(hex_color: str) -> Tuple[int, ...]:
    """Convert hex color to RGBA tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 8:
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4, 6))
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (*rgb, 255)


def compose_layers(layers: list, width: int = THUMB_W, height: int = THUMB_H) -> dict:
    """
    Compose multiple image layers into a single thumbnail.

    Each layer dict:
      asset_id or path, x, y, scale, opacity, rotation, flip_h, anchor
    Layers rendered bottom-to-top (first = background).
    """
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    print(f"  Canvas: {width}x{height}, {len(layers)} layer(s)")

    # Update manifest with composition
    manifest = get_manifest()
    manifest["composition"] = {"layers": layers, "width": width, "height": height}
    save_manifest(manifest)

    for i, layer in enumerate(layers):
        try:
            # Resolve image path
            if "asset_id" in layer:
                img_path = resolve_asset_path(layer["asset_id"])
            elif "path" in layer:
                img_path = Path(layer["path"])
            else:
                print(f"  WARNING: Layer {i} has no asset_id or path, skipping")
                continue

            img = Image.open(img_path).convert("RGBA")

            # Scale
            scale = layer.get("scale", 1.0)
            if scale != 1.0:
                new_w = max(1, int(img.width * scale))
                new_h = max(1, int(img.height * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Flip
            if layer.get("flip_h", False):
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Rotation
            rotation = layer.get("rotation", 0)
            if rotation != 0:
                img = img.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)

            # Opacity
            opacity = layer.get("opacity", 1.0)
            if opacity < 1.0:
                alpha = img.getchannel("A")
                alpha = alpha.point(lambda a: int(a * opacity))
                img.putalpha(alpha)

            # Position with anchor
            x = layer.get("x", 0)
            y = layer.get("y", 0)
            anchor = layer.get("anchor", "top_left")
            x, y = _apply_anchor(x, y, img.width, img.height, anchor)

            # Paste onto canvas
            canvas.paste(img, (x, y), img)
            name = layer.get("asset_id", layer.get("path", f"layer_{i}"))
            print(f"    Layer {i}: {name} at ({x},{y}) scale={scale}")

        except Exception as e:
            print(f"  ERROR on layer {i}: {e}")

    # Save composed result
    out_path = WORKSPACE_DIR / "composed.png"
    canvas.convert("RGB").save(str(out_path), "PNG", quality=95)
    # Also save RGBA version for further compositing
    canvas.save(str(WORKSPACE_DIR / "composed_rgba.png"), "PNG")
    print(f"  Saved: composed.png ({width}x{height})")

    return {"path": str(out_path), "width": width, "height": height, "layers": len(layers)}


def add_text(
    target: str,
    text: str,
    x: int,
    y: int,
    font_name: str = "Montserrat-ExtraBold.ttf",
    font_size: int = 72,
    color: str = "#FFFFFF",
    highlight_color: str = "#FFD700",
    highlight_uppercase: bool = True,
    stroke_width: int = 4,
    stroke_color: str = "#000000",
    shadow: bool = True,
    shadow_color: str = "#00000080",
    shadow_offset: tuple = (3, 3),
    max_width: int = 0,
    align: str = "left",
    highlight_bg_color: str = "",
    highlight_bg_padding: int = 8,
    highlight_bg_radius: int = 4,
) -> dict:
    """
    Add text overlay with advanced styling.
    UPPERCASE words are automatically highlighted in highlight_color.
    """
    if target in ("composed", "composed.png"):
        img_path = WORKSPACE_DIR / "composed.png"
    else:
        img_path = resolve_asset_path(target)

    if not img_path.exists():
        raise FileNotFoundError(f"Target not found: {img_path}")

    img = Image.open(img_path).convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)
    font = get_font(font_name, font_size)

    text_rgba = hex_to_rgba(color)
    hl_rgba = hex_to_rgba(highlight_color)
    stroke_rgba = hex_to_rgba(stroke_color)
    shadow_rgba = hex_to_rgba(shadow_color)

    # Word wrap
    if max_width > 0:
        lines = _wrap_text(text, font, max_width, draw)
    else:
        lines = text.split("\n")

    # Measure line height
    ref_bbox = draw.textbbox((0, 0), "Mg", font=font)
    line_h = ref_bbox[3] - ref_bbox[1]
    line_spacing = int(line_h * 0.15)

    current_y = y

    for line in lines:
        words = line.split()
        if not words:
            current_y += line_h + line_spacing
            continue

        # Calculate line width for alignment
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_w = line_bbox[2] - line_bbox[0]

        if align == "center":
            line_x = x - line_w // 2
        elif align == "right":
            line_x = x - line_w
        else:
            line_x = x

        word_x = line_x
        for word in words:
            is_upper = word.isupper() and highlight_uppercase and len(word) > 1
            word_color = hl_rgba[:3] if is_upper else text_rgba[:3]

            # Background pill for highlighted words
            if is_upper and highlight_bg_color:
                hbg_rgba = hex_to_rgba(highlight_bg_color)
                word_bbox = draw.textbbox((word_x, current_y), word, font=font)
                pill_x1 = word_bbox[0] - highlight_bg_padding
                pill_y1 = word_bbox[1] - highlight_bg_padding
                pill_x2 = word_bbox[2] + highlight_bg_padding
                pill_y2 = word_bbox[3] + highlight_bg_padding
                _draw_rounded_rect(draw, pill_x1, pill_y1, pill_x2, pill_y2,
                                   fill=hbg_rgba, radius=highlight_bg_radius)

            # Shadow
            if shadow:
                draw.text(
                    (word_x + shadow_offset[0], current_y + shadow_offset[1]),
                    word, font=font, fill=shadow_rgba,
                )

            # Stroke + fill
            draw.text(
                (word_x, current_y), word, font=font,
                fill=(*word_color, 255),
                stroke_width=stroke_width,
                stroke_fill=stroke_rgba,
            )

            # Advance cursor
            bbox = draw.textbbox((0, 0), word + " ", font=font)
            word_x += bbox[2] - bbox[0]

        current_y += line_h + line_spacing

    result = Image.alpha_composite(img, txt_layer)

    # Save
    out_path = WORKSPACE_DIR / "composed.png"
    result.convert("RGB").save(str(out_path), "PNG", quality=95)
    result.save(str(WORKSPACE_DIR / "composed_rgba.png"), "PNG")
    print(f"  Text added: \"{text[:50]}\" at ({x},{y})")
    return {"path": str(out_path), "text": text}


def refine_layer(action: str, layer_index: int = 0, params: dict = None) -> dict:
    """
    Make targeted adjustments to the composition.
    Actions: move_layer, scale_layer, opacity_layer, remove_layer, crop
    """
    manifest = get_manifest()
    layers = manifest.get("composition", {}).get("layers", [])
    params = params or {}

    if action == "remove_layer":
        if 0 <= layer_index < len(layers):
            removed = layers.pop(layer_index)
            manifest["composition"]["layers"] = layers
            save_manifest(manifest)
            # Re-render
            w = manifest["composition"].get("width", THUMB_W)
            h = manifest["composition"].get("height", THUMB_H)
            return compose_layers(layers, w, h)
        return {"error": f"Layer index {layer_index} out of range (0-{len(layers)-1})"}

    if action in ("move_layer", "scale_layer", "opacity_layer"):
        if 0 <= layer_index < len(layers):
            if action == "move_layer":
                layers[layer_index]["x"] = params.get("x", layers[layer_index].get("x", 0))
                layers[layer_index]["y"] = params.get("y", layers[layer_index].get("y", 0))
            elif action == "scale_layer":
                layers[layer_index]["scale"] = params.get("scale", 1.0)
            elif action == "opacity_layer":
                layers[layer_index]["opacity"] = params.get("opacity", 1.0)

            manifest["composition"]["layers"] = layers
            save_manifest(manifest)
            w = manifest["composition"].get("width", THUMB_W)
            h = manifest["composition"].get("height", THUMB_H)
            return compose_layers(layers, w, h)
        return {"error": f"Layer index {layer_index} out of range"}

    if action == "crop":
        img_path = WORKSPACE_DIR / "composed.png"
        if not img_path.exists():
            return {"error": "No composed image to crop"}
        img = Image.open(img_path)
        cx = params.get("x", 0)
        cy = params.get("y", 0)
        cw = params.get("w", img.width)
        ch = params.get("h", img.height)
        cropped = img.crop((cx, cy, cx + cw, cy + ch))
        cropped.save(str(img_path), "PNG", quality=95)
        return {"path": str(img_path), "crop": [cx, cy, cw, ch]}

    return {"error": f"Unknown action: {action}"}


# ── Helpers ────────────────────────────────────────────────────────────

def _apply_anchor(x: int, y: int, w: int, h: int, anchor: str) -> Tuple[int, int]:
    """Convert anchor-relative coords to top-left coords."""
    if anchor == "center":
        return x - w // 2, y - h // 2
    elif anchor == "bottom_center":
        return x - w // 2, y - h
    elif anchor == "top_center":
        return x - w // 2, y
    elif anchor == "bottom_left":
        return x, y - h
    elif anchor == "bottom_right":
        return x - w, y - h
    elif anchor == "top_right":
        return x - w, y
    return x, y  # top_left default


def _wrap_text(text: str, font, max_width: int, draw) -> List[str]:
    """Wrap text into lines fitting within max_width."""
    segments = text.split("\n")
    lines = []
    for segment in segments:
        words = segment.split()
        if not words:
            lines.append("")
            continue
        current = []
        for word in words:
            test = " ".join(current + [word])
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
    return lines


def _draw_rounded_rect(
    draw: ImageDraw.Draw,
    x1: int, y1: int, x2: int, y2: int,
    fill: tuple,
    radius: int = 4,
) -> None:
    """Draw a rectangle with rounded corners on the given draw context."""
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        draw.rectangle([x1, y1, x2, y2], fill=fill)
        return
    draw.rectangle([x1 + r, y1, x2 - r, y2], fill=fill)
    draw.rectangle([x1, y1 + r, x2, y2 - r], fill=fill)
    draw.pieslice([x1, y1, x1 + 2*r, y1 + 2*r], 180, 270, fill=fill)
    draw.pieslice([x2 - 2*r, y1, x2, y1 + 2*r], 270, 360, fill=fill)
    draw.pieslice([x1, y2 - 2*r, x1 + 2*r, y2], 90, 180, fill=fill)
    draw.pieslice([x2 - 2*r, y2 - 2*r, x2, y2], 0, 90, fill=fill)
