#!/usr/bin/env python3
"""
Helamans Videos - MCP Server for Claude Desktop
================================================
Model Context Protocol server that exposes all video production tools
as MCP tools for Claude Desktop integration.

Tools:
  - check_project_status: Check current project state
  - archive_project: Archive current project and prepare workspace
  - save_script: Save generated script JSON to current.json
  - validate_script: Validate script against all production rules
  - generate_images: Generate images with Gemini 2.5 Flash
  - generate_audio: Generate narration audio with ElevenLabs V3
  - render_video: Render final video with Ken Burns + lower third
  - generate_thumbnail: Generate YouTube thumbnail (with layout presets!)
  - configure_thumbnail: Update thumbnail_config.json settings
  - list_thumbnail_layouts: List available thumbnail layout presets
  - get_metadata: Get video metadata (title, description, tags, hashtags)
  - list_archives: List all archived projects
  - get_prompt_template: Get the script generation prompt template
  - get_video_ideas_prompt: Get the video ideas generation prompt

Run: python mcp_server.py
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Any

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = DATA_DIR / "scripts"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_DIR = DATA_DIR / "output"
MUSIC_DIR = DATA_DIR / "music"
ARCHIVE_DIR = DATA_DIR / "archive"
CONFIG_PATH = DATA_DIR / "config.json"
THUMBNAIL_CONFIG_PATH = DATA_DIR / "thumbnail_config.json"
PROMPT_TEMPLATE_PATH = DATA_DIR / "PROMPT_TEMPLATE.txt"
PROMPT_IDEAS_PATH = DATA_DIR / "PROMPT_VIDEO_IDEAS.txt"

# Ensure directories exist
for d in [SCRIPTS_DIR, IMAGES_DIR, AUDIO_DIR, OUTPUT_DIR, ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Helpers ───────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    """Load JSON file with BOM handling."""
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()
    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No valid JSON in {path}")
    return json.loads(content[start:end + 1])


def save_json(path: Path, data: dict) -> None:
    """Save JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def deep_merge(base: dict, update: dict) -> None:
    """Deep merge update into base dict, modifying base in place."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def count_files(directory: Path, extensions: list[str] = None) -> int:
    if not directory.exists():
        return 0
    if extensions:
        return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in extensions)
    return sum(1 for f in directory.iterdir() if f.is_file())


# ─── Script Validation ────────────────────────────────────────────────

ALLOWED_TAGS = ["[softly]", "[pause]", "[pensive]", "[warmly]", "[hopeful]", "[awe]"]
FORBIDDEN_TAGS = ["[reverently]"]
VALID_MUSIC = ["divinity.mp3", "mistery.mp3", "revelations.mp3"]


def validate_script_data(script_json: dict) -> dict:
    """
    Validate script JSON against all production rules.
    Returns dict with 'valid' (bool), 'errors' (list), 'warnings' (list).
    """
    errors = []
    warnings = []

    script = script_json.get("script", {})
    if not script:
        return {"valid": False, "errors": ["Missing 'script' root key"], "warnings": []}

    # ID check
    if script.get("id") != "current":
        errors.append("script.id must be 'current'")

    # Titles
    title_yt = script.get("title_youtube", "")
    title_int = script.get("title_internal", "")
    if not title_yt:
        errors.append("Missing title_youtube")
    elif len(title_yt) > 70:
        errors.append(f"title_youtube too long ({len(title_yt)} chars, max 70)")
    if not title_int:
        errors.append("Missing title_internal")
    elif len(title_int) > 80:
        errors.append(f"title_internal too long ({len(title_int)} chars, max 80)")

    # Thumbnail asset
    thumb_asset = script.get("thumbnail_asset_id", "")
    if not thumb_asset:
        warnings.append("Missing thumbnail_asset_id (recommended: '1a' - the image designed for thumbnail)")
    else:
        valid_ids = [f"1{chr(97+i)}" for i in range(14)]  # 1a through 1n
        if thumb_asset not in valid_ids:
            warnings.append(f"thumbnail_asset_id '{thumb_asset}' is not a valid asset ID")

    # Music
    music = script.get("background_music", "")
    if music not in VALID_MUSIC:
        errors.append(f"Invalid background_music '{music}'. Must be one of: {VALID_MUSIC}")

    # Narration
    narration = script.get("narration", {})
    full_text = narration.get("full_text", "")

    if not full_text:
        errors.append("Missing narration.full_text")
    else:
        # Ellipsis checks
        if full_text.startswith("..."):
            errors.append("full_text starts with '...' (causes garbled audio)")
        if full_text.endswith("..."):
            errors.append("full_text ends with '...' (causes trailing garble)")

        # Question checks
        if full_text.strip().startswith("¿") or full_text.strip().startswith("?"):
            errors.append("full_text starts with a question (breaks loop technique)")
        stripped = full_text.rstrip()
        if stripped and stripped[-1] == '?':
            errors.append("full_text ends with a question (breaks loop technique)")

        # Forbidden tags
        for tag in FORBIDDEN_TAGS:
            if tag in full_text:
                errors.append(f"Forbidden tag {tag} found (distorts voice)")

        # Word count
        clean_text = re.sub(r'\[.*?\]', '', full_text)
        word_count = len(clean_text.split())
        if word_count < 350:
            warnings.append(f"Narration has {word_count} words (recommended 350-450 for 2-3 min)")
        elif word_count > 500:
            warnings.append(f"Narration has {word_count} words (may exceed 3 min)")

        # Tag count
        tags_found = re.findall(r'\[.*?\]', full_text)
        tag_count = len(tags_found)
        if tag_count > 12:
            warnings.append(f"Too many audio tags ({tag_count}, max 12)")

        # Check text length for tag activation
        if tag_count > 0 and len(full_text) < 250:
            warnings.append("Text is under 250 chars - audio tags may not activate in ElevenLabs V3")

        # Duplicate start/end check
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        if len(sentences) >= 3:
            def strip_tags(s):
                return re.sub(r'\[.*?\]', '', s).strip().lower()
            first = strip_tags(sentences[0])
            last = strip_tags(sentences[-1])
            if first and last:
                first_words = set(first.split())
                last_words = set(last.split())
                if first_words and last_words:
                    overlap = len(first_words & last_words) / max(len(first_words), len(last_words))
                    if overlap > 0.7:
                        errors.append("First and last sentences are too similar (broken loop)")

    # Visual assets
    visual_assets = narration.get("visual_assets", [])
    if len(visual_assets) < 10:
        errors.append(f"Too few visual_assets ({len(visual_assets)}, min 10)")
    elif len(visual_assets) > 14:
        errors.append(f"Too many visual_assets ({len(visual_assets)}, max 14)")

    # Check word index continuity
    if visual_assets:
        prev_end = -1
        for i, asset in enumerate(visual_assets):
            start_idx = asset.get("start_word_index", -1)
            end_idx = asset.get("end_word_index", -1)
            asset_id = asset.get("visual_asset_id", f"asset_{i}")

            if i == 0 and start_idx != 0:
                errors.append(f"First asset '{asset_id}' start_word_index is {start_idx}, must be 0")
            elif i > 0 and start_idx != prev_end + 1:
                errors.append(f"Gap in word indices between asset {i-1} and '{asset_id}': expected {prev_end + 1}, got {start_idx}")

            if end_idx < start_idx:
                errors.append(f"Asset '{asset_id}' end_word_index ({end_idx}) < start_word_index ({start_idx})")

            prev_end = end_idx

            # Image prompt checks
            prompt = asset.get("image_prompt", "")
            if not prompt:
                errors.append(f"Asset '{asset_id}' missing image_prompt")
            else:
                if "Cinematic oil painting" in prompt:
                    errors.append(f"Asset '{asset_id}' uses 'Cinematic oil painting' (must use 'Latter-day Saint sacred art')")
                if "ANGELS WITHOUT WINGS" in prompt:
                    errors.append(f"Asset '{asset_id}' says 'ANGELS WITHOUT WINGS' (must say 'NO WINGS on angels')")
                if "NO CROSSES" not in prompt or "NO HALOS" not in prompt:
                    warnings.append(f"Asset '{asset_id}' image_prompt missing 'NO CROSSES, NO HALOS' ending")
                if "16:9" not in prompt:
                    warnings.append(f"Asset '{asset_id}' image_prompt missing '16:9' specification")

    # Metadata
    metadata = script.get("metadata", {})
    desc_hook = metadata.get("description_hook", "")
    desc_body = metadata.get("description_body", "")
    desc_bullets = metadata.get("description_bullets", [])
    escrituras = metadata.get("escrituras_mencionadas", [])
    tags = metadata.get("tags", [])
    hashtags = metadata.get("hashtags", [])

    # Support legacy single "description" field
    legacy_desc = metadata.get("description", "")
    if legacy_desc and not desc_hook:
        warnings.append("Using legacy 'description' field. New scripts should use 'description_hook' + 'description_body' + 'description_bullets'")

    if not desc_hook and not legacy_desc:
        errors.append("Missing metadata.description_hook (or legacy metadata.description)")
    elif desc_hook and len(desc_hook) > 150:
        warnings.append(f"description_hook has {len(desc_hook)} chars (max 150 recommended for 'mostrar más' visibility)")

    if not desc_body and not legacy_desc:
        warnings.append("Missing metadata.description_body")

    if len(desc_bullets) < 3 and not legacy_desc:
        warnings.append(f"Only {len(desc_bullets)} description_bullets (recommended 3-5)")

    if len(escrituras) < 1 and not legacy_desc:
        warnings.append("Missing metadata.escrituras_mencionadas")

    if len(tags) < 20:
        warnings.append(f"Only {len(tags)} tags (recommended 30-40 search phrases)")
    if len(hashtags) < 5:
        warnings.append(f"Only {len(hashtags)} hashtags (recommended 5)")
    if "#RelatosDeLuz" not in hashtags:
        warnings.append("Missing #RelatosDeLuz hashtag")
    if "#VenSigueme" not in hashtags and "#VenSígueme" not in hashtags:
        warnings.append("Missing #VenSigueme hashtag")

    # Check title has search context (pipe separator)
    title_yt = script.get("title_youtube", "")
    if title_yt and "|" not in title_yt:
        warnings.append("title_youtube missing ' | search context' suffix for SEO (e.g. 'Hook Title | Topic Libro de Mormón')")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


# ─── MCP Server ───────────────────────────────────────────────────────

app = Server("helamans-videos")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="check_project_status",
            description=(
                "Check the current project status. Returns what phase the project is in "
                "(empty, script, images, audio, rendered, complete) and what files exist. "
                "CALL THIS FIRST to understand the workspace state."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="archive_project",
            description=(
                "Archive the current project to data/archive/{timestamp}/ and clear workspace. "
                "Use this before starting a new video to preserve previous work."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm archiving",
                        "default": False,
                    }
                },
                "required": ["confirm"],
            },
        ),
        Tool(
            name="save_script",
            description=(
                "Save a generated script JSON to data/scripts/current.json. "
                "The script MUST follow the exact schema with script.id='current', "
                "title_youtube, title_internal, background_music, narration.full_text, "
                "narration.visual_assets[], and metadata. "
                "Automatically validates before saving."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "script_json": {
                        "type": "object",
                        "description": "The complete script JSON object with 'script' root key",
                    },
                    "escritura": {
                        "type": "string",
                        "description": "Scripture reference to update in config.json (e.g. 'Moisés 8:19-30')",
                    },
                },
                "required": ["script_json"],
            },
        ),
        Tool(
            name="validate_script",
            description=(
                "Validate the current script at data/scripts/current.json against all "
                "production rules. Returns errors and warnings. Use this before generating "
                "images or audio."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="generate_images",
            description=(
                "Generate all images for the current script using Gemini 2.5 Flash Image. "
                "Images are 16:9 horizontal format with LDS sacred art styling enforced. "
                "Skips images that already exist unless force=true. "
                "Requires: data/scripts/current.json + GEMINI_API_KEY in .env"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Regenerate all images even if they exist",
                        "default": False,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="generate_audio",
            description=(
                "Generate narration audio with ElevenLabs V3 and word-level timestamps with Whisper. "
                "Automatically sanitizes the narration text (strips ellipsis, removes [reverently], "
                "detects duplicate sentences). "
                "Requires: data/scripts/current.json + ELEVEN_LABS_API_KEY in .env"
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="align_timestamps",
            description=(
                "CRITICAL STEP: Run this AFTER generate_audio and BEFORE render_video. "
                "Uses content-based matching to find where each visual asset's narration "
                "appears in the Whisper transcript, then sets start_time/end_time (seconds) "
                "directly on each asset. This eliminates fragile word-index remapping. "
                "Use auto_fix=true to save the corrected script, or false for a dry-run report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_fix": {
                        "type": "boolean",
                        "description": "If true, automatically save the realigned script. If false, just report.",
                        "default": False,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="render_video",
            description=(
                "Render the final video (1920x1080 H.264 MP4) with Ken Burns effect, "
                "professional lower third overlay, and mixed audio (narration + background music). "
                "Uses NVIDIA NVENC GPU acceleration or falls back to libx264 CPU. "
                "Requires: script + images + audio + aligned timestamps all ready first."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="generate_thumbnail",
            description=(
                "Generate a professional YouTube thumbnail (1280x720 PNG). "
                "Choose a LAYOUT PRESET for visual variety! Available layouts: "
                "luminoso (bright/hopeful), dramatico (dark/intense), celestial (elegant/sacred), "
                "profeta (split/character focus), esperanza (warm/devotional), "
                "impacto (bold/modern), minimalista (clean/quiet). "
                "Each layout changes gradient, text position, fonts, colors, and accents. "
                "Use list_thumbnail_layouts to see details. "
                "Pass overrides (asset_id, custom_title, zoom, pan) on top of any layout."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "layout": {
                        "type": "string",
                        "description": (
                            "Layout preset name for visual variety. One of: luminoso, dramatico, "
                            "celestial, profeta, esperanza, impacto, minimalista. "
                            "Each produces a dramatically different thumbnail style."
                        ),
                    },
                    "asset_id": {
                        "type": "string",
                        "description": "Visual asset ID to use as image (e.g. '1a', '1c'). Default: auto-select.",
                    },
                    "custom_title": {
                        "type": "string",
                        "description": "Short 3-5 word title for thumbnail (overrides config). Use UPPERCASE for gold words.",
                    },
                    "zoom": {
                        "type": "number",
                        "description": "Zoom level 1.0-1.3 (default 1.05)",
                    },
                    "pan_x": {
                        "type": "number",
                        "description": "Horizontal pan -0.5 to 0.5 (default 0.0)",
                    },
                    "pan_y": {
                        "type": "number",
                        "description": "Vertical pan -0.5 to 0.5 (default 0.15)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="configure_thumbnail",
            description=(
                "Update thumbnail_config.json settings. Use this to fine-tune the thumbnail "
                "before generating: select image, adjust zoom/pan, set custom title, change colors."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "config_updates": {
                        "type": "object",
                        "description": "Partial thumbnail config to merge (deep merge into existing config)",
                    }
                },
                "required": ["config_updates"],
            },
        ),
        Tool(
            name="list_thumbnail_layouts",
            description=(
                "List all available thumbnail layout presets with names, descriptions, "
                "and mood tags. Use this to choose the right layout for a video's tone/theme. "
                "Each layout produces a dramatically different thumbnail style for visual variety."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_metadata",
            description=(
                "Get the video metadata from the current script: YouTube title, description, "
                "tags, and hashtags. Ready to paste into YouTube Studio."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="list_archives",
            description="List all archived projects with their timestamps.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_prompt_template",
            description=(
                "Get the PROMPT_TEMPLATE.txt content for script generation. "
                "This is the master prompt that defines storytelling rules, audio tags, "
                "visual asset requirements, and JSON schema. "
                "Use this as the system prompt when generating a new script."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tema": {
                        "type": "string",
                        "description": "Topic/theme for the video",
                    },
                    "escritura": {
                        "type": "string",
                        "description": "Scripture reference(s)",
                    },
                    "contexto": {
                        "type": "string",
                        "description": "Additional context, story details, historical facts",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_video_ideas_prompt",
            description=(
                "Get the PROMPT_VIDEO_IDEAS.txt template for generating 20 video ideas "
                "from Ven Sigueme scripture text. Returns the prompt ready to use."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ven_sigueme_text": {
                        "type": "string",
                        "description": "The Ven Sigueme scripture text to analyze for video ideas",
                    }
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        result = await _handle_tool(name, arguments)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"ERROR: {str(e)}")]


async def _handle_tool(name: str, args: dict[str, Any]) -> str:

    # ── check_project_status ──────────────────────────────────────
    if name == "check_project_status":
        status = {"phase": "empty", "files": {}}

        # Script
        script_path = SCRIPTS_DIR / "current.json"
        has_script = file_exists(script_path)
        status["files"]["script"] = str(script_path) if has_script else None

        # Images
        img_count = count_files(IMAGES_DIR, [".png", ".jpg", ".jpeg", ".webp"])
        status["files"]["images"] = img_count

        # Audio
        has_audio = file_exists(AUDIO_DIR / "current.mp3")
        has_timestamps = file_exists(AUDIO_DIR / "current_timestamps.json")
        status["files"]["audio"] = str(AUDIO_DIR / "current.mp3") if has_audio else None
        status["files"]["timestamps"] = str(AUDIO_DIR / "current_timestamps.json") if has_timestamps else None

        # Output
        has_video = file_exists(OUTPUT_DIR / "current.mp4")
        has_thumbnail = file_exists(OUTPUT_DIR / "thumbnail.png")
        has_metadata = file_exists(OUTPUT_DIR / "metadata.txt")
        status["files"]["video"] = str(OUTPUT_DIR / "current.mp4") if has_video else None
        status["files"]["thumbnail"] = str(OUTPUT_DIR / "thumbnail.png") if has_thumbnail else None
        status["files"]["metadata"] = str(OUTPUT_DIR / "metadata.txt") if has_metadata else None

        # Determine phase
        if has_video and has_thumbnail:
            status["phase"] = "complete"
        elif has_video:
            status["phase"] = "rendered"
        elif has_audio and has_timestamps:
            status["phase"] = "audio_ready"
        elif img_count > 0:
            status["phase"] = "images_ready"
        elif has_script:
            status["phase"] = "script_ready"

        # Script details
        if has_script:
            try:
                script = load_json(script_path)
                sd = script.get("script", {})
                status["script_info"] = {
                    "title_youtube": sd.get("title_youtube", ""),
                    "title_internal": sd.get("title_internal", ""),
                    "background_music": sd.get("background_music", ""),
                    "visual_assets_count": len(sd.get("narration", {}).get("visual_assets", [])),
                }
            except Exception:
                status["script_info"] = "Error reading script"

        # Next steps
        next_steps = {
            "empty": "Generate a script using get_prompt_template, then save with save_script",
            "script_ready": "Generate images with generate_images",
            "images_ready": "Generate audio with generate_audio",
            "audio_ready": "IMPORTANT: Run align_timestamps to verify/fix word indices BEFORE rendering",
            "rendered": "Generate thumbnail with generate_thumbnail",
            "complete": "Project is complete! Archive with archive_project before starting new one",
        }
        status["next_step"] = next_steps.get(status["phase"], "")

        return json.dumps(status, indent=2, ensure_ascii=False)

    # ── archive_project ───────────────────────────────────────────
    elif name == "archive_project":
        if not args.get("confirm"):
            return "ERROR: Must pass confirm=true to archive. This will move all current files to archive."

        from src.archive_manager import archive_current_project, clear_working_directories, has_current_project

        if not has_current_project():
            return "No current project to archive. Workspace is already clean."

        archive_path = archive_current_project()
        clear_working_directories()
        return f"Project archived to: {archive_path}\nWorkspace cleared and ready for new project."

    # ── save_script ───────────────────────────────────────────────
    elif name == "save_script":
        script_json = args.get("script_json")
        if not script_json:
            return "ERROR: script_json is required"

        # Validate first
        validation = validate_script_data(script_json)
        if not validation["valid"]:
            error_list = "\n".join(f"  - {e}" for e in validation["errors"])
            warning_list = "\n".join(f"  - {w}" for w in validation["warnings"])
            msg = f"VALIDATION FAILED - Fix these errors before saving:\n\nERRORS:\n{error_list}"
            if validation["warnings"]:
                msg += f"\n\nWARNINGS:\n{warning_list}"
            return msg

        # Save script
        script_path = SCRIPTS_DIR / "current.json"
        save_json(script_path, script_json)

        # Update config.json with escritura if provided
        escritura = args.get("escritura")
        if escritura and file_exists(CONFIG_PATH):
            config = load_json(CONFIG_PATH)
            if "video_metadata" not in config:
                config["video_metadata"] = {}
            config["video_metadata"]["escritura"] = escritura
            save_json(CONFIG_PATH, config)

        result = f"Script saved to: {script_path}"
        if validation["warnings"]:
            warning_list = "\n".join(f"  - {w}" for w in validation["warnings"])
            result += f"\n\nWARNINGS (non-blocking):\n{warning_list}"
        if escritura:
            result += f"\n\nConfig updated with escritura: {escritura}"

        return result

    # ── validate_script ───────────────────────────────────────────
    elif name == "validate_script":
        script_path = SCRIPTS_DIR / "current.json"
        if not file_exists(script_path):
            return "ERROR: No script found at data/scripts/current.json"

        script_json = load_json(script_path)
        validation = validate_script_data(script_json)

        if validation["valid"]:
            msg = "SCRIPT IS VALID - Ready for image/audio generation."
        else:
            error_list = "\n".join(f"  - {e}" for e in validation["errors"])
            msg = f"VALIDATION FAILED:\n\nERRORS:\n{error_list}"

        if validation["warnings"]:
            warning_list = "\n".join(f"  - {w}" for w in validation["warnings"])
            msg += f"\n\nWARNINGS:\n{warning_list}"

        return msg

    # ── generate_images ───────────────────────────────────────────
    elif name == "generate_images":
        from src.image_generator import generate_all_images, check_images_exist

        force = args.get("force", False)

        existing, missing = check_images_exist()
        if not missing and not force:
            return f"All {len(existing)} images already exist: {', '.join(existing)}\nUse force=true to regenerate."

        success, total = generate_all_images(force=force)

        # Check final state
        existing_after, missing_after = check_images_exist()

        result = f"Image generation complete: {success}/{total} successful"
        if existing_after:
            result += f"\nExisting: {', '.join(existing_after)}"
        if missing_after:
            result += f"\nMISSING: {', '.join(missing_after)} - retry or generate manually"

        return result

    # ── generate_audio ────────────────────────────────────────────
    elif name == "generate_audio":
        from src.audio_generator import load_script, sanitize_narration, generate_audio, generate_timestamps, load_config

        config = load_config()
        script = load_script()

        narration = script.get("script", {}).get("narration", {})
        text = narration.get("full_text", "")
        if not text:
            return "ERROR: No narration text found in script"

        text = sanitize_narration(text)
        voice_id = config.get("voice_id", "YqZLNYWZm98oKaaLZkUA")

        # Save sanitized text for alignment reference
        sanitized_path = AUDIO_DIR / "current_sanitized.txt"
        with open(sanitized_path, "w", encoding="utf-8") as f:
            f.write(text)

        audio_path = AUDIO_DIR / "current.mp3"
        timestamps_path = AUDIO_DIR / "current_timestamps.json"

        generate_audio(text, voice_id, audio_path)
        timestamps = generate_timestamps(audio_path, timestamps_path)

        return (
            f"Audio generated successfully!\n"
            f"  Audio: {audio_path}\n"
            f"  Duration: {timestamps['duration']:.1f} seconds\n"
            f"  Words: {len(timestamps['words'])}\n"
            f"  Timestamps: {timestamps_path}\n\n"
            f"NEXT STEP: Run align_timestamps to verify word indices match before rendering."
        )

    # ── align_timestamps ──────────────────────────────────────────
    elif name == "align_timestamps":
        if not file_exists(SCRIPTS_DIR / "current.json"):
            return "ERROR: No script found at data/scripts/current.json"
        if not file_exists(AUDIO_DIR / "current_timestamps.json"):
            return "ERROR: No timestamps found. Run generate_audio first."

        from src.timestamp_aligner import align_and_fix

        auto_fix = args.get("auto_fix", False)
        report, remapped = align_and_fix(auto_save=auto_fix)

        return report

    # ── render_video ──────────────────────────────────────────────
    elif name == "render_video":
        from src.video_renderer import render_video as do_render

        output_path = OUTPUT_DIR / "current.mp4"
        do_render(output_path)

        if file_exists(output_path):
            size_mb = output_path.stat().st_size / (1024 * 1024)
            return (
                f"Video rendered successfully!\n"
                f"  Output: {output_path}\n"
                f"  Size: {size_mb:.1f} MB\n"
                f"  Metadata saved to: {OUTPUT_DIR / 'metadata.txt'}"
            )
        return "ERROR: Video render completed but output file not found"

    # ── generate_thumbnail ────────────────────────────────────────
    elif name == "generate_thumbnail":
        layout_name = args.get("layout")

        # Start with existing config or empty
        thumb_cfg = load_json(THUMBNAIL_CONFIG_PATH) if file_exists(THUMBNAIL_CONFIG_PATH) else {"thumbnail": {}}

        # If layout is specified, apply it as the base
        if layout_name:
            from src.thumbnail_layouts import get_layout_config, LAYOUTS
            if layout_name not in LAYOUTS:
                available = ", ".join(LAYOUTS.keys())
                return f"ERROR: Unknown layout '{layout_name}'. Available: {available}"

            layout_config = get_layout_config(layout_name)
            deep_merge(thumb_cfg, layout_config)
            # Store layout name for logging
            thumb_cfg.setdefault("thumbnail", {})["_layout_name"] = layout_name

        tc = thumb_cfg.get("thumbnail", {})

        # Apply explicit overrides on top of layout
        if "asset_id" in args:
            tc.setdefault("image", {})["source"] = "asset"
            tc["image"]["asset_id"] = args["asset_id"]
        if "custom_title" in args:
            tc.setdefault("title", {})["text"] = "custom"
            tc["title"]["custom_text"] = args["custom_title"]
        if "zoom" in args:
            tc.setdefault("image", {})["zoom"] = args["zoom"]
        if "pan_x" in args:
            tc.setdefault("image", {})["pan_x"] = args["pan_x"]
        if "pan_y" in args:
            tc.setdefault("image", {})["pan_y"] = args["pan_y"]

        thumb_cfg["thumbnail"] = tc
        save_json(THUMBNAIL_CONFIG_PATH, thumb_cfg)

        from src.thumbnail_generator import generate_thumbnail as do_thumbnail
        output = do_thumbnail()

        result = f"Thumbnail generated: {output}"
        if layout_name:
            result += f"\nLayout used: {layout_name}"
        return result

    # ── configure_thumbnail ───────────────────────────────────────
    elif name == "configure_thumbnail":
        config_updates = args.get("config_updates", {})
        if not config_updates:
            return "ERROR: config_updates is required"

        # Load existing config
        if file_exists(THUMBNAIL_CONFIG_PATH):
            existing = load_json(THUMBNAIL_CONFIG_PATH)
        else:
            existing = {"thumbnail": {}}

        deep_merge(existing, config_updates)
        save_json(THUMBNAIL_CONFIG_PATH, existing)

        return f"Thumbnail config updated:\n{json.dumps(existing, indent=2, ensure_ascii=False)}"

    # ── list_thumbnail_layouts ─────────────────────────────────────
    elif name == "list_thumbnail_layouts":
        from src.thumbnail_layouts import list_layouts
        layouts = list_layouts()
        lines = ["AVAILABLE THUMBNAIL LAYOUTS:", ""]
        for layout in layouts:
            tags = ", ".join(layout["mood_tags"][:5])
            lines.append(f"  {layout['id'].upper()} ({layout['name']})")
            lines.append(f"    {layout['description']}")
            lines.append(f"    Mood tags: {tags}")
            lines.append("")
        lines.append("Usage: generate_thumbnail(layout='esperanza', custom_title='MI TÍTULO')")
        return "\n".join(lines)

    # ── get_metadata ──────────────────────────────────────────────
    elif name == "get_metadata":
        script_path = SCRIPTS_DIR / "current.json"
        if not file_exists(script_path):
            return "ERROR: No script found"

        script = load_json(script_path)
        sd = script.get("script", {})
        metadata = sd.get("metadata", {})

        title_youtube = sd.get("title_youtube", "")
        hashtags = metadata.get("hashtags", [])
        tags = metadata.get("tags", [])
        desc_hook = metadata.get("description_hook", "")
        desc_body = metadata.get("description_body", "")
        desc_bullets = metadata.get("description_bullets", [])
        escrituras = metadata.get("escrituras_mencionadas", [])
        legacy_desc = metadata.get("description", "")

        sep = "=" * 45
        line = "─" * 30

        parts = []
        parts.append(sep)
        parts.append("YOUTUBE TITLE:")
        parts.append(sep)
        parts.append(title_youtube)

        parts.append(sep)
        parts.append("DESCRIPTION:")
        parts.append(sep)

        if desc_hook:
            parts.append(desc_hook)
            parts.append("")
            if desc_body:
                parts.append(desc_body)
                parts.append("")
            parts.append("En este video exploramos:")
            for bullet in desc_bullets:
                parts.append(bullet)
            parts.append("")
            parts.append(line)
            if escrituras:
                parts.append("📌 RECURSOS MENCIONADOS:")
                for esc in escrituras:
                    parts.append(f"• {esc}")
                parts.append(line)
                parts.append("")
            parts.append("Si este mensaje te tocó el corazón, compártelo con alguien que necesite escucharlo. 💛")
            parts.append("")
            parts.append("🔔 SUSCRÍBETE a Relatos de Luz para más historias del Evangelio que fortalecen tu fe cada semana.")
            parts.append("💬 Deja tu comentario contándonos tu experiencia.")
            parts.append("👍 Dale LIKE si quieres más videos como este.")
            parts.append("")
            parts.append(line)
            parts.append("📱 Síguenos:")
            parts.append("Canal: @RelatosDeLuz")
            parts.append(line)
            parts.append("")
            parts.append(" ".join(hashtags))
        else:
            parts.append(legacy_desc)
            parts.append("")
            parts.append(" ".join(hashtags))

        parts.append("")
        parts.append(sep)
        parts.append("TAGS (copiar y pegar en YouTube Studio):")
        parts.append(sep)
        parts.append(",".join(tags))

        parts.append("")
        parts.append(sep)
        parts.append("HASHTAGS (ya incluidos en descripción):")
        parts.append(sep)
        parts.append(" ".join(hashtags))

        parts.append("")
        parts.append(sep)
        parts.append("NOTAS SEO:")
        parts.append(sep)
        if "|" in title_youtube:
            parts.append("- El título usa HOOK + contexto de búsqueda separado por '|'")
        parts.append("- Tags incluyen variaciones con y sin acentos para capturar más búsquedas")
        parts.append("- Las primeras 2 líneas de la descripción son visibles antes del 'mostrar más'")
        parts.append("- Emojis en descripción mejoran CTR")

        return "\n".join(parts)

    # ── list_archives ─────────────────────────────────────────────
    elif name == "list_archives":
        if not ARCHIVE_DIR.exists():
            return "No archive directory found."

        archives = sorted(
            [d for d in ARCHIVE_DIR.iterdir() if d.is_dir()],
            reverse=True
        )

        if not archives:
            return "No archived projects."

        lines = ["ARCHIVED PROJECTS:", ""]
        for a in archives:
            files = list(a.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            lines.append(f"  {a.name}  ({file_count} files)")

        return "\n".join(lines)

    # ── get_prompt_template ───────────────────────────────────────
    elif name == "get_prompt_template":
        if not file_exists(PROMPT_TEMPLATE_PATH):
            return "ERROR: PROMPT_TEMPLATE.txt not found"

        with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = f.read()

        # Fill in values if provided
        tema = args.get("tema", "[TEMA]")
        escritura = args.get("escritura", "[ESCRITURA]")
        contexto = args.get("contexto", "[CONTEXTO]")

        template = template.replace("[TEMA]", tema)
        template = template.replace("[ESCRITURA]", escritura)
        template = template.replace("[CONTEXTO]", contexto)

        return template

    # ── get_video_ideas_prompt ────────────────────────────────────
    elif name == "get_video_ideas_prompt":
        if not file_exists(PROMPT_IDEAS_PATH):
            return "ERROR: PROMPT_VIDEO_IDEAS.txt not found"

        with open(PROMPT_IDEAS_PATH, "r", encoding="utf-8") as f:
            template = f.read()

        ven_sigueme_text = args.get("ven_sigueme_text", "")
        if ven_sigueme_text:
            template += "\n" + ven_sigueme_text

        return template

    else:
        return f"ERROR: Unknown tool '{name}'"


# ─── Main ─────────────────────────────────────────────────────────────

async def main():
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
