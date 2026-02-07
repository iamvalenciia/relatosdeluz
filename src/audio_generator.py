"""
Audio Generator using ElevenLabs API V3
Generates narration audio from script and creates word-level timestamps with Whisper.
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = DATA_DIR / "scripts"
AUDIO_DIR = DATA_DIR / "audio"
CONFIG_PATH = DATA_DIR / "config.json"

# Tags that are known to distort the voice - strip them before sending to ElevenLabs
FORBIDDEN_TAGS = ["[reverently]"]

# Tags that are allowed
ALLOWED_TAGS = ["[softly]", "[pause]", "[pensive]", "[warmly]", "[hopeful]", "[awe]"]


def sanitize_narration(text: str) -> str:
    """
    Sanitize narration text before sending to ElevenLabs.
    Fixes common script generation errors that cause audio glitches.

    Fixes applied:
    1. Strip leading "..." (causes garbled start)
    2. Strip trailing "..." (causes garbled trailing syllable)
    3. Remove forbidden tags like [reverently] (distorts voice)
    4. Ensure text ends with clean punctuation (not ellipsis)
    5. Remove duplicate start/end text (broken loop attempts)
    """
    original = text
    fixes = []

    # 1. Strip leading "..."
    if text.startswith("..."):
        text = text[3:].lstrip()
        fixes.append("Stripped leading '...'")

    # 2. Strip trailing "..."
    if text.endswith("..."):
        text = text[:-3].rstrip()
        fixes.append("Stripped trailing '...'")

    # Also catch "?..." or "!..." patterns at the end
    text = re.sub(r'([.!?])\.\.\.$', r'\1', text)
    text = re.sub(r'\.\.\.\s*$', '', text)

    # 3. Remove forbidden tags
    for tag in FORBIDDEN_TAGS:
        if tag in text:
            text = text.replace(tag, "")
            fixes.append(f"Removed forbidden tag {tag}")

    # 4. Clean up double spaces left by tag removal
    text = re.sub(r'  +', ' ', text)

    # 5. Ensure text ends with clean punctuation
    text = text.rstrip()
    if text and text[-1] not in '.!?':
        text += '.'
        fixes.append("Added period at end for clean audio cutoff")

    # 6. Detect if the last sentence is a near-duplicate of the first sentence
    # (broken loop where AI repeated the opening line at the end)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) >= 3:
        # Clean tags for comparison
        def strip_tags(s):
            return re.sub(r'\[.*?\]', '', s).strip().lower()

        first = strip_tags(sentences[0])
        last = strip_tags(sentences[-1])

        # Check similarity - if last sentence is >70% similar to first, remove it
        if first and last:
            # Simple word overlap check
            first_words = set(first.split())
            last_words = set(last.split())
            if first_words and last_words:
                overlap = len(first_words & last_words) / max(len(first_words), len(last_words))
                if overlap > 0.7:
                    text = '. '.join(sentences[:-1])
                    if not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
                        text += '.'
                    fixes.append(f"Removed duplicate ending ('{sentences[-1][:40]}...')")

    if fixes:
        print("\n  SANITIZER FIXES APPLIED:")
        for fix in fixes:
            print(f"    - {fix}")
        print()

    return text.strip()


def load_config() -> dict:
    """Load configuration from config.json"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_script() -> dict:
    """Load the current script from data/scripts/current.json"""
    script_path = SCRIPTS_DIR / "current.json"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found at {script_path}. "
            "Please create your script JSON and save it as data/scripts/current.json"
        )
    # Read with utf-8-sig to automatically handle BOM
    with open(script_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()
    
    # Handle case where Claude CLI adds extra text before/after JSON
    # Find the JSON object boundaries
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No valid JSON object found in script file")
    
    json_content = content[start_idx:end_idx + 1]
    return json.loads(json_content)


def generate_audio(text: str, voice_id: str, output_path: Path) -> None:
    """
    Generate audio using ElevenLabs API V3.
    
    Args:
        text: The narration text to convert to speech
        voice_id: ElevenLabs voice ID
        output_path: Path to save the MP3 file
    """
    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVEN_LABS_API_KEY not found in environment variables")
    
    client = ElevenLabs(api_key=api_key)
    
    print(f"Generating audio with voice ID: {voice_id}")
    print(f"Text length: {len(text)} characters")
    
    # Generate audio using ElevenLabs V3 (eleven_v3 supports audio tags like [softly], [pause], etc.)
    # Note: Audio tags require prompts longer than 250 characters for best results
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_v3",  # V3 model supports audio tags
        output_format="mp3_44100_192"  # Higher quality: 192kbps
    )
    
    # Write audio to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)
    
    print(f"Audio saved to: {output_path}")


def generate_timestamps(audio_path: Path, output_path: Path) -> dict:
    """
    Generate word-level timestamps using Whisper.
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the timestamps JSON
        
    Returns:
        Dictionary with word-level timestamps
    """
    import whisper
    
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    print("Transcribing audio for timestamps...")
    result = model.transcribe(
        str(audio_path),
        language="es",
        word_timestamps=True
    )
    
    # Extract word-level timestamps
    timestamps = {
        "segments": [],
        "words": [],
        "duration": result.get("duration", 0)
    }
    
    word_index = 0
    for segment in result.get("segments", []):
        seg_data = {
            "id": segment.get("id"),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": segment.get("text"),
            "words": []
        }
        
        for word_info in segment.get("words", []):
            word_data = {
                "word": word_info.get("word", "").strip(),
                "start": word_info.get("start"),
                "end": word_info.get("end"),
                "index": word_index
            }
            seg_data["words"].append(word_data)
            timestamps["words"].append(word_data)
            word_index += 1
        
        timestamps["segments"].append(seg_data)
    
    # Save timestamps
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(timestamps, f, ensure_ascii=False, indent=2)
    
    print(f"Timestamps saved to: {output_path}")
    print(f"Total words: {len(timestamps['words'])}")
    print(f"Audio duration: {timestamps['duration']:.2f} seconds")
    
    return timestamps


def main():
    """Main entry point for audio generation."""
    print("=" * 50)
    print("HELAMANS VIDEOS - Audio Generator")
    print("=" * 50)
    
    # Ensure audio directory exists
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load config and script
    config = load_config()
    script = load_script()
    
    # Extract narration text
    narration = script.get("script", {}).get("narration", {})
    text = narration.get("full_text", "")

    if not text:
        raise ValueError("No narration text found in script. Check 'script.narration.full_text'")

    # Sanitize text to prevent audio glitches
    text = sanitize_narration(text)

    voice_id = config.get("voice_id", "YqZLNYWZm98oKaaLZkUA")

    # Paths
    audio_path = AUDIO_DIR / "current.mp3"
    timestamps_path = AUDIO_DIR / "current_timestamps.json"

    # Generate audio
    generate_audio(text, voice_id, audio_path)
    
    # Generate timestamps
    generate_timestamps(audio_path, timestamps_path)
    
    print("\n" + "=" * 50)
    print("Audio generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
