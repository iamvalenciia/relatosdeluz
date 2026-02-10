"""
Timestamp Aligner - Realigns visual_assets word indices to match Whisper timestamps.

The Problem:
  1. Script full_text has word indices based on Python str.split()
  2. Sanitizer modifies text before ElevenLabs (removes tags, ellipsis, duplicates)
  3. Whisper transcribes the AUDIO and assigns its own word indices
  4. Script word indices ≠ Whisper word indices → image timing breaks

The Solution:
  1. Split the sanitized narration into "script words" (excluding audio tags)
  2. Fuzzy-match each script word to the closest Whisper word by text similarity
  3. Build a mapping: script_word_index → whisper_word_index
  4. Remap visual_assets start/end_word_index using this mapping
  5. Validate that each image covers the right narration segment
"""

import re
import json
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = DATA_DIR / "scripts"
AUDIO_DIR = DATA_DIR / "audio"


def load_script() -> dict:
    path = SCRIPTS_DIR / "current.json"
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()
    start = content.find('{')
    end = content.rfind('}')
    return json.loads(content[start:end + 1])


def load_timestamps() -> dict:
    path = AUDIO_DIR / "current_timestamps.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_audio_tags(text: str) -> str:
    """Remove audio tags like [softly], [pause], etc."""
    return re.sub(r'\[.*?\]', '', text).strip()


def normalize_word(word: str) -> str:
    """Normalize a word for comparison: lowercase, strip punctuation."""
    return re.sub(r'[^\w]', '', word.lower())


def tokenize_script_text(full_text: str) -> list[dict]:
    """
    Tokenize the script full_text into words, skipping audio tags.
    Returns list of {index, word, original} where index is the position
    in the tag-stripped word list (matching what Whisper would hear).
    """
    # Remove audio tags first
    clean = strip_audio_tags(full_text)
    # Normalize whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    words = clean.split()

    return [{"index": i, "word": w, "normalized": normalize_word(w)} for i, w in enumerate(words)]


def build_alignment(script_words: list[dict], whisper_words: list[dict]) -> list[dict]:
    """
    Build word-by-word alignment between script words and Whisper words.

    Uses sequential matching: walk through both lists in order,
    allowing skips for mismatches (Whisper may split/merge words).

    Returns list of {script_idx, whisper_idx, script_word, whisper_word, confidence}.
    """
    alignment = []
    w_idx = 0  # Current position in whisper words

    for s in script_words:
        s_norm = s["normalized"]
        if not s_norm:
            continue

        best_match = None
        best_score = 0.0
        best_w_idx = None

        # Search forward in whisper words within a window
        # Look ahead up to 5 positions to handle insertions/deletions
        search_end = min(w_idx + 8, len(whisper_words))
        # Also look back 2 positions for edge cases
        search_start = max(0, w_idx - 2)

        for wi in range(search_start, search_end):
            w = whisper_words[wi]
            w_norm = normalize_word(w.get("word", ""))
            if not w_norm:
                continue

            # Exact match
            if s_norm == w_norm:
                best_match = w
                best_score = 1.0
                best_w_idx = wi
                break

            # Fuzzy match (handles Whisper minor differences)
            score = SequenceMatcher(None, s_norm, w_norm).ratio()
            if score > best_score and score >= 0.6:
                best_match = w
                best_score = score
                best_w_idx = wi

        if best_match and best_w_idx is not None:
            alignment.append({
                "script_idx": s["index"],
                "whisper_idx": best_match["index"],
                "script_word": s["word"],
                "whisper_word": best_match.get("word", ""),
                "confidence": best_score,
                "whisper_start": best_match.get("start", 0.0),
                "whisper_end": best_match.get("end", 0.0),
            })
            w_idx = best_w_idx + 1
        else:
            # No match found — record gap
            alignment.append({
                "script_idx": s["index"],
                "whisper_idx": None,
                "script_word": s["word"],
                "whisper_word": None,
                "confidence": 0.0,
                "whisper_start": None,
                "whisper_end": None,
            })

    return alignment


def remap_visual_assets(
    visual_assets: list[dict],
    alignment: list[dict],
    whisper_word_count: int,
) -> tuple[list[dict], list[dict]]:
    """
    Remap visual_assets word indices from script-based to Whisper-based.

    Returns (remapped_assets, changes_log).
    """
    # Build lookup: script_idx → whisper_idx
    idx_map = {}
    for a in alignment:
        if a["whisper_idx"] is not None:
            idx_map[a["script_idx"]] = a["whisper_idx"]

    remapped = []
    changes = []

    for asset in visual_assets:
        asset_id = asset.get("visual_asset_id", "?")
        old_start = asset.get("start_word_index", 0)
        old_end = asset.get("end_word_index", 0)

        # Find closest mapped index for start
        new_start = _find_closest_mapped(old_start, idx_map, direction="forward")
        # Find closest mapped index for end
        new_end = _find_closest_mapped(old_end, idx_map, direction="backward")

        # Safety: clamp to valid range
        if new_start is not None:
            new_start = max(0, new_start)
        else:
            new_start = old_start  # fallback

        if new_end is not None:
            new_end = min(new_end, whisper_word_count - 1)
        else:
            new_end = old_end  # fallback

        # Ensure end >= start
        if new_end < new_start:
            new_end = new_start

        new_asset = dict(asset)
        new_asset["start_word_index"] = new_start
        new_asset["end_word_index"] = new_end
        remapped.append(new_asset)

        if old_start != new_start or old_end != new_end:
            changes.append({
                "asset_id": asset_id,
                "old_range": f"{old_start}-{old_end}",
                "new_range": f"{new_start}-{new_end}",
            })

    # Fix continuity: ensure no gaps between assets
    for i in range(1, len(remapped)):
        prev_end = remapped[i - 1]["end_word_index"]
        curr_start = remapped[i]["start_word_index"]
        if curr_start != prev_end + 1:
            remapped[i]["start_word_index"] = prev_end + 1
            if remapped[i]["start_word_index"] > remapped[i]["end_word_index"]:
                remapped[i]["end_word_index"] = remapped[i]["start_word_index"]

    return remapped, changes


def _find_closest_mapped(target_idx: int, idx_map: dict, direction: str = "forward") -> Optional[int]:
    """Find the whisper index for the closest mapped script index."""
    if target_idx in idx_map:
        return idx_map[target_idx]

    # Search nearby indices
    max_search = 10
    if direction == "forward":
        for offset in range(1, max_search):
            if target_idx + offset in idx_map:
                return idx_map[target_idx + offset]
            if target_idx - offset in idx_map:
                return idx_map[target_idx - offset]
    else:  # backward
        for offset in range(1, max_search):
            if target_idx - offset in idx_map:
                return idx_map[target_idx - offset]
            if target_idx + offset in idx_map:
                return idx_map[target_idx + offset]

    return None


def generate_alignment_report(
    alignment: list[dict],
    visual_assets: list[dict],
    remapped_assets: list[dict],
    changes: list[dict],
    whisper_words: list[dict],
) -> str:
    """Generate a human-readable alignment report."""
    lines = []
    lines.append("=" * 70)
    lines.append("TIMESTAMP ALIGNMENT REPORT")
    lines.append("=" * 70)

    # Summary stats
    total = len(alignment)
    matched = sum(1 for a in alignment if a["confidence"] >= 0.8)
    fuzzy = sum(1 for a in alignment if 0.6 <= a["confidence"] < 0.8)
    missing = sum(1 for a in alignment if a["confidence"] == 0.0)

    lines.append(f"\nWord Alignment: {matched}/{total} exact, {fuzzy} fuzzy, {missing} missing")
    lines.append(f"Script words: {total}")
    lines.append(f"Whisper words: {len(whisper_words)}")

    # Show low-confidence matches
    low_conf = [a for a in alignment if 0.0 < a["confidence"] < 0.8]
    if low_conf:
        lines.append("\nFUZZY MATCHES (review these):")
        for a in low_conf:
            lines.append(
                f"  script[{a['script_idx']}] '{a['script_word']}' → "
                f"whisper[{a['whisper_idx']}] '{a['whisper_word']}' "
                f"(confidence: {a['confidence']:.0%})"
            )

    # Show unmatched
    unmatched = [a for a in alignment if a["confidence"] == 0.0]
    if unmatched:
        lines.append("\nUNMATCHED WORDS (no Whisper equivalent):")
        for a in unmatched:
            lines.append(f"  script[{a['script_idx']}] '{a['script_word']}'")

    # Show changes
    lines.append("\n" + "-" * 70)
    lines.append("VISUAL ASSET REMAPPING:")
    lines.append("-" * 70)

    if not changes:
        lines.append("  No changes needed - indices already aligned!")
    else:
        for c in changes:
            lines.append(f"  {c['asset_id']}: {c['old_range']} → {c['new_range']}")

    # Show final asset timing
    lines.append("\nFINAL ASSET TIMING:")
    for asset in remapped_assets:
        aid = asset["visual_asset_id"]
        s = asset["start_word_index"]
        e = asset["end_word_index"]

        # Find timing from whisper
        start_time = None
        end_time = None
        for w in whisper_words:
            if w["index"] == s and start_time is None:
                start_time = w.get("start", 0.0)
            if w["index"] == e:
                end_time = w.get("end", 0.0)

        time_str = ""
        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            time_str = f"  {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)"
        else:
            time_str = "  MISSING TIMESTAMP DATA"

        # Show narration summary snippet
        summary = asset.get("narration_summary", "")[:50]
        lines.append(f"  {aid}: words {s}-{e}{time_str}")
        if summary:
            lines.append(f"         \"{summary}...\"")

    # Check for gaps
    lines.append("\nCONTINUITY CHECK:")
    has_gaps = False
    for i in range(1, len(remapped_assets)):
        prev_end = remapped_assets[i - 1]["end_word_index"]
        curr_start = remapped_assets[i]["start_word_index"]
        if curr_start != prev_end + 1:
            lines.append(f"  GAP: {remapped_assets[i-1]['visual_asset_id']} ends at {prev_end}, "
                         f"{remapped_assets[i]['visual_asset_id']} starts at {curr_start}")
            has_gaps = True
    if not has_gaps:
        lines.append("  All assets are contiguous - no gaps!")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def align_and_fix(auto_save: bool = False) -> tuple[str, list[dict]]:
    """
    Main alignment function.

    1. Load script and timestamps
    2. Tokenize script text (excluding audio tags)
    3. Build word alignment with Whisper
    4. Remap visual_assets indices
    5. Generate report
    6. Optionally save fixed script

    Returns (report_text, remapped_assets).
    """
    script = load_script()
    timestamps = load_timestamps()

    sd = script.get("script", {})
    narration = sd.get("narration", {})
    full_text = narration.get("full_text", "")
    visual_assets = narration.get("visual_assets", [])
    whisper_words = timestamps.get("words", [])

    if not full_text:
        return "ERROR: No narration text in script", []
    if not whisper_words:
        return "ERROR: No words in timestamps (run generate_audio first)", []
    if not visual_assets:
        return "ERROR: No visual_assets in script", []

    # Step 1: Tokenize script text (same as what ElevenLabs received after sanitizer)
    script_words = tokenize_script_text(full_text)

    # Step 2: Build alignment
    alignment = build_alignment(script_words, whisper_words)

    # Step 3: Remap visual assets
    remapped, changes = remap_visual_assets(visual_assets, alignment, len(whisper_words))

    # Step 4: Report
    report = generate_alignment_report(alignment, visual_assets, remapped, changes, whisper_words)

    # Step 5: Auto-save if requested
    if auto_save and changes:
        narration["visual_assets"] = remapped
        script_path = SCRIPTS_DIR / "current.json"
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        report += f"\n\nScript SAVED with realigned indices to {script_path}"

    return report, remapped


def main():
    """CLI entry point."""
    import sys
    auto_save = "--fix" in sys.argv

    print("TIMESTAMP ALIGNER")
    print("=" * 50)

    if auto_save:
        print("Mode: AUTO-FIX (will save changes)")
    else:
        print("Mode: DRY-RUN (use --fix to save changes)")

    print()

    report, _ = align_and_fix(auto_save=auto_save)
    print(report)


if __name__ == "__main__":
    main()
