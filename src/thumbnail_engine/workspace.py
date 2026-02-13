"""
Workspace Manager - Tracks generated assets and composition state.

The workspace is a directory (data/thumbnail_workspace/) that holds all
generated images, a manifest.json tracking every asset, and the current
composition state (layer order, positions, etc.).
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
WORKSPACE_DIR = DATA_DIR / "thumbnail_workspace"
MANIFEST_PATH = WORKSPACE_DIR / "manifest.json"
OUTPUT_DIR = DATA_DIR / "output"


def init_workspace() -> str:
    """Create or clear workspace and manifest. Returns workspace path."""
    if WORKSPACE_DIR.exists():
        for f in WORKSPACE_DIR.iterdir():
            if f.is_file():
                f.unlink()
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "assets": {},
        "composition": {"layers": []},
    }
    save_manifest(manifest)
    return str(WORKSPACE_DIR)


def get_manifest() -> dict:
    """Load and return the current manifest."""
    if not MANIFEST_PATH.exists():
        init_workspace()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    """Save the manifest to disk."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def next_id(prefix: str) -> str:
    """Generate next available ID like bg_01, bg_02, el_01..."""
    manifest = get_manifest()
    existing = [k for k in manifest.get("assets", {}) if k.startswith(prefix)]
    if not existing:
        return f"{prefix}01"
    nums = [int(k.replace(prefix, "")) for k in existing]
    return f"{prefix}{max(nums) + 1:02d}"


def add_asset(asset_id: str, asset_type: str, path: Path, prompt: str = "") -> dict:
    """Register an asset in the manifest. Returns asset info."""
    manifest = get_manifest()
    try:
        img = Image.open(path)
        w, h = img.size
    except Exception:
        w, h = 0, 0

    info = {
        "type": asset_type,
        "path": path.name,
        "width": w,
        "height": h,
        "prompt": prompt,
        "created_at": datetime.now().isoformat(),
    }
    manifest.setdefault("assets", {})[asset_id] = info
    save_manifest(manifest)
    return info


def list_assets() -> list:
    """List all assets in workspace with metadata."""
    manifest = get_manifest()
    result = []
    for aid, info in manifest.get("assets", {}).items():
        result.append({"id": aid, **info})
    return result


def resolve_asset_path(asset_id: str) -> Path:
    """Resolve an asset ID to its file path. Also accepts absolute paths."""
    p = Path(asset_id)
    if p.is_absolute() and p.exists():
        return p
    # Check workspace
    manifest = get_manifest()
    asset = manifest.get("assets", {}).get(asset_id)
    if asset:
        return WORKSPACE_DIR / asset["path"]
    # Try as filename in workspace
    candidate = WORKSPACE_DIR / asset_id
    if candidate.exists():
        return candidate
    # Try video images directory
    images_dir = DATA_DIR / "images"
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = images_dir / f"{asset_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Asset not found: {asset_id}")


def finalize_thumbnail(source: str = "composed.png") -> Path:
    """Copy the final thumbnail to data/output/thumbnail.png."""
    if source in ("composed", "composed.png"):
        src = WORKSPACE_DIR / "composed.png"
    elif source in ("final", "final.png"):
        src = WORKSPACE_DIR / "final.png"
    else:
        src = resolve_asset_path(source)

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / "thumbnail.png"
    shutil.copy2(str(src), str(dest))
    return dest


def cleanup_workspace() -> str:
    """Remove all workspace files."""
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return "Workspace cleaned"
