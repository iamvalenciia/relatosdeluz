"""
Timestamp Aligner - Content-based alignment of visual assets to audio timestamps.

The Problem:
  1. Script full_text has word indices based on Python str.split() (excluding audio tags)
  2. Sanitizer modifies text before ElevenLabs (removes tags, ellipsis, duplicates)
  3. Whisper transcribes the AUDIO and assigns its own word indices
  4. Script word indices ≠ Whisper word indices → image timing breaks

The Solution (Content-Based Matching):
  Instead of mapping individual words, we match CHUNKS of narration text:
  1. For each visual asset, extract its narration text from the script
  2. Find that text in the Whisper transcript using sliding-window matching
  3. Write start_time/end_time (seconds) directly on each asset
  4. The renderer uses time ranges directly — no word-index lookups needed

  This is robust against: Whisper word merging/splitting, AI miscounting
  indices, sanitizer text modifications, and Spanish accent differences.
"""

import re
import json
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

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


def build_clean_word_list(full_text: str) -> list[str]:
    """
    Strip all audio tags from full_text, normalize whitespace,
    and return a flat list of clean words.
    This produces the word list the AI was supposed to use when counting indices.
    """
    clean = strip_audio_tags(full_text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean.split()


def extract_asset_narration(clean_words: list[str], start_idx: int, end_idx: int) -> str:
    """
    Extract the narration text for a visual asset from the clean word list.
    Clamps indices to valid range.
    """
    start = max(0, start_idx)
    end = min(len(clean_words) - 1, end_idx)
    if start > end:
        return ''
    return ' '.join(clean_words[start:end + 1])


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for fuzzy substring matching:
    - lowercase
    - strip all punctuation except spaces
    - collapse whitespace
    - strip accents (e.g., 'días' -> 'dias')
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Strip accents
    nfkd = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_asset_time_range(
    asset_text: str,
    whisper_words: list[dict],
    search_start_idx: int = 0,
    min_confidence: float = 0.45,
) -> tuple:
    """
    Find where asset_text appears in the Whisper transcript using
    sliding-window normalized text matching.

    Strategy:
    1. Normalize asset text and whisper words
    2. Build candidate windows of varying word count
    3. Score each window using SequenceMatcher on normalized text
    4. Return the best match's time range

    Args:
        asset_text: The narration text for this visual asset
        whisper_words: Full list of Whisper word dicts
        search_start_idx: Whisper index to start searching from
        min_confidence: Minimum match score to accept

    Returns:
        (whisper_start_idx, whisper_end_idx, start_time, end_time, confidence)
    """
    norm_asset = normalize_for_matching(asset_text)
    asset_word_count = len(norm_asset.split())

    if asset_word_count == 0 or not whisper_words:
        return None, None, 0.0, 0.0, 0.0

    # Window size range: accommodate Whisper splitting/merging words
    min_window = max(1, asset_word_count - 5)
    max_window = asset_word_count + 5

    # Search constraint: assets are sequential, content should be nearby
    # Allow generous forward search but not infinite
    search_end = min(len(whisper_words), search_start_idx + asset_word_count * 3 + 15)

    best_score = 0.0
    best_window = None  # (start_idx, end_idx)

    # Phase 1: Coarse search (stride 3) to find approximate region
    coarse_best_start = search_start_idx
    coarse_best_score = 0.0
    coarse_window = asset_word_count  # use exact word count for coarse

    for start in range(search_start_idx, max(search_start_idx, search_end - coarse_window + 1), 3):
        end = min(start + coarse_window - 1, len(whisper_words) - 1)
        window_text = ' '.join(w.get("word", "") for w in whisper_words[start:end + 1])
        norm_window = normalize_for_matching(window_text)

        if not norm_window:
            continue

        score = SequenceMatcher(None, norm_asset, norm_window).ratio()
        if score > coarse_best_score:
            coarse_best_score = score
            coarse_best_start = start

    # Phase 2: Fine search around the coarse result
    fine_search_start = max(search_start_idx, coarse_best_start - 6)
    fine_search_end = min(len(whisper_words), coarse_best_start + asset_word_count + 12)

    for window_size in range(min_window, max_window + 1):
        for start in range(fine_search_start, max(fine_search_start, fine_search_end - window_size + 1)):
            end = start + window_size - 1
            if end >= len(whisper_words):
                break

            window_text = ' '.join(w.get("word", "") for w in whisper_words[start:end + 1])
            norm_window = normalize_for_matching(window_text)

            if not norm_window:
                continue

            score = SequenceMatcher(None, norm_asset, norm_window).ratio()

            if score > best_score:
                best_score = score
                best_window = (start, end)

                # Early exit on near-perfect match
                if score > 0.95:
                    break

        if best_score > 0.95:
            break

    if best_window and best_score >= min_confidence:
        s_idx, e_idx = best_window
        start_time = whisper_words[s_idx].get("start", 0.0)
        end_time = whisper_words[e_idx].get("end", 0.0)
        return s_idx, e_idx, start_time, end_time, best_score

    # Fallback: no match found
    return None, None, 0.0, 0.0, 0.0


def fix_timing_gaps(results: list[dict], audio_duration: float) -> None:
    """
    Ensure perfect contiguous timing coverage:
    1. First asset starts at 0.0
    2. Each asset's start_time = previous asset's end_time
    3. Last asset's end_time = audio_duration
    No gaps, no overlaps.

    Modifies results in place.
    """
    if not results:
        return

    # Force first asset to start at time 0
    results[0]["start_time"] = 0.0

    # Force contiguity: each start = previous end
    for i in range(1, len(results)):
        results[i]["start_time"] = results[i - 1]["end_time"]

    # Force last asset to extend to audio end
    results[-1]["end_time"] = audio_duration


def generate_report(results: list[dict], clean_words: list[str],
                    whisper_words: list[dict], audio_duration: float) -> str:
    """Generate a human-readable diagnostic report."""
    lines = []
    lines.append("=" * 70)
    lines.append("TIMESTAMP ALIGNMENT REPORT (Content-Based)")
    lines.append("=" * 70)

    lines.append(f"\nScript words: {len(clean_words)} (after stripping audio tags)")
    lines.append(f"Whisper words: {len(whisper_words)}")
    lines.append(f"Audio duration: {audio_duration:.1f}s")

    # Asset timing
    lines.append("\nASSET TIMING:")
    low_confidence = []
    for r in results:
        asset = r["asset"]
        aid = asset.get("visual_asset_id", "?")
        st = r["start_time"]
        et = r["end_time"]
        dur = et - st
        conf = r["confidence"]

        lines.append(f"  {aid}: {st:.3f}s - {et:.3f}s ({dur:.1f}s) [confidence: {conf:.2f}]")

        # Show text snippets
        script_snippet = r.get("asset_text_snippet", "")
        if script_snippet:
            lines.append(f"      Script:  \"{script_snippet}...\"")

        whisper_snippet = r.get("whisper_text_snippet", "")
        if whisper_snippet:
            lines.append(f"      Whisper: \"{whisper_snippet}...\"")

        if conf < 0.7:
            low_confidence.append(aid)

    # Warnings
    lines.append("")
    if low_confidence:
        lines.append("WARNINGS:")
        for aid in low_confidence:
            lines.append(f"  Asset {aid} has low confidence - review manually")
    else:
        lines.append("WARNINGS: None")

    # Continuity check
    lines.append("\nCONTINUITY CHECK:")
    has_gaps = False
    for i in range(1, len(results)):
        prev_end = results[i - 1]["end_time"]
        curr_start = results[i]["start_time"]
        diff = abs(curr_start - prev_end)
        if diff > 0.01:  # tolerance of 10ms
            aid_prev = results[i - 1]["asset"].get("visual_asset_id", "?")
            aid_curr = results[i]["asset"].get("visual_asset_id", "?")
            lines.append(f"  GAP: {aid_prev} ends at {prev_end:.3f}s, "
                         f"{aid_curr} starts at {curr_start:.3f}s (gap: {diff:.3f}s)")
            has_gaps = True
    if not has_gaps:
        lines.append("  All assets are contiguous - perfect timing coverage!")

    # Duration sanity
    lines.append("\nDURATION SANITY CHECK:")
    durations = [(r["asset"].get("visual_asset_id", "?"), r["end_time"] - r["start_time"]) for r in results]
    if durations:
        shortest = min(durations, key=lambda x: x[1])
        longest = max(durations, key=lambda x: x[1])
        lines.append(f"  Shortest: {shortest[0]} at {shortest[1]:.1f}s {'(WARNING: very short!)' if shortest[1] < 1.0 else '(OK)'}")
        lines.append(f"  Longest: {longest[0]} at {longest[1]:.1f}s {'(WARNING: very long!)' if longest[1] > 40.0 else '(OK)'}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def align_and_fix(auto_save: bool = False) -> tuple[str, list[dict]]:
    """
    Main alignment function - content-based approach.

    1. Load script and timestamps
    2. Build clean word list from full_text (strip audio tags)
    3. For each visual asset:
       a. Extract its narration text using the script's word indices
       b. Find that text chunk in the Whisper transcript via sliding window
       c. Record start_time and end_time
    4. Fix timing gaps (ensure contiguous coverage)
    5. Generate diagnostic report
    6. Optionally save start_time/end_time on each asset

    Returns (report_text, updated_assets).
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

    audio_duration = timestamps.get("duration", 0.0)
    if audio_duration == 0.0 and whisper_words:
        audio_duration = whisper_words[-1].get("end", 0.0)

    # Step 1: Build clean word list (same counting basis the AI used)
    clean_words = build_clean_word_list(full_text)

    # Step 2: Process each asset sequentially
    search_start_idx = 0
    results = []

    for asset in visual_assets:
        asset_text = extract_asset_narration(
            clean_words,
            asset.get("start_word_index", 0),
            asset.get("end_word_index", 0),
        )

        w_start, w_end, start_time, end_time, confidence = find_asset_time_range(
            asset_text, whisper_words, search_start_idx
        )

        # Build whisper text snippet for report
        whisper_snippet = ""
        if w_start is not None and w_end is not None:
            whisper_snippet = ' '.join(
                w.get("word", "") for w in whisper_words[w_start:min(w_start + 10, w_end + 1)]
            )

        results.append({
            "asset": asset,
            "asset_text_snippet": asset_text[:80] if asset_text else "",
            "whisper_text_snippet": whisper_snippet[:80],
            "start_time": start_time,
            "end_time": end_time,
            "confidence": confidence,
            "whisper_start_idx": w_start,
            "whisper_end_idx": w_end,
        })

        # Advance search window for next asset
        if w_end is not None:
            search_start_idx = w_end + 1

    # Step 3: Handle any unmatched assets with proportional fallback
    _apply_proportional_fallback(results, audio_duration, clean_words, visual_assets)

    # Step 4: Fix timing gaps for perfect contiguous coverage
    fix_timing_gaps(results, audio_duration)

    # Step 5: Generate report
    report = generate_report(results, clean_words, whisper_words, audio_duration)

    # Step 6: Write start_time/end_time into assets and optionally save
    for r in results:
        r["asset"]["start_time"] = round(r["start_time"], 3)
        r["asset"]["end_time"] = round(r["end_time"], 3)

    if auto_save:
        narration["visual_assets"] = [r["asset"] for r in results]
        script_path = SCRIPTS_DIR / "current.json"
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        report += f"\n\nScript SAVED with time ranges to {script_path}"

    return report, [r["asset"] for r in results]


def _apply_proportional_fallback(
    results: list[dict],
    audio_duration: float,
    clean_words: list[str],
    visual_assets: list[dict],
) -> None:
    """
    For any assets where content matching failed (confidence=0),
    estimate their time range proportionally based on word count.
    """
    total_words = len(clean_words)
    if total_words == 0:
        return

    # Find matched anchors to interpolate from
    matched = [(i, r) for i, r in enumerate(results) if r["confidence"] > 0]

    for i, r in enumerate(results):
        if r["confidence"] > 0:
            continue

        asset = r["asset"]
        asset_words = asset.get("end_word_index", 0) - asset.get("start_word_index", 0) + 1
        word_fraction = asset_words / total_words

        # Find nearest matched neighbors
        prev_end_time = 0.0
        next_start_time = audio_duration

        for j in range(i - 1, -1, -1):
            if results[j]["confidence"] > 0:
                prev_end_time = results[j]["end_time"]
                break

        for j in range(i + 1, len(results)):
            if results[j]["confidence"] > 0:
                next_start_time = results[j]["start_time"]
                break

        # Distribute available time proportionally
        available = next_start_time - prev_end_time
        # Count unmatched assets in this gap
        gap_assets = []
        for j in range(i, len(results)):
            if results[j]["confidence"] > 0:
                break
            gap_assets.append(j)
        # Go backward too
        for j in range(i - 1, -1, -1):
            if results[j]["confidence"] > 0:
                break
            if j not in gap_assets:
                gap_assets.insert(0, j)

        gap_total_words = sum(
            results[j]["asset"].get("end_word_index", 0) - results[j]["asset"].get("start_word_index", 0) + 1
            for j in gap_assets
        )

        # Distribute
        current_time = prev_end_time
        for j in gap_assets:
            a = results[j]["asset"]
            w = a.get("end_word_index", 0) - a.get("start_word_index", 0) + 1
            fraction = w / gap_total_words if gap_total_words > 0 else 1.0 / len(gap_assets)
            duration = available * fraction
            results[j]["start_time"] = current_time
            results[j]["end_time"] = current_time + duration
            results[j]["confidence"] = 0.1  # Mark as estimated
            current_time += duration


def main():
    """CLI entry point."""
    import sys
    auto_save = "--fix" in sys.argv

    print("TIMESTAMP ALIGNER (Content-Based)")
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
