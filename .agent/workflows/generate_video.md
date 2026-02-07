---
description: Generate a complete Ven Sígueme video from topic, scripture, and context
---

# Video Generation Workflow

This workflow creates Ven Sígueme videos with storytelling-driven, gospel-centered content for the Relatos de Luz channel.

// turbo-all

## Prerequisites
- Claude CLI installed
- ElevenLabs API key configured in `.env` (ELEVEN_LABS_API_KEY)
- Gemini API key configured in `.env` (GEMINI_API_KEY)

## Steps

### 1. Check for Existing Project
```bash
python -m src.archive_manager
```
If project exists, choose to archive or continue.

### 2. Collect Video Information
Ask the user to provide:
- **Tema**: Topic title
- **Escritura**: Scripture reference
- **Contexto**: Story context (real experiences, scripture references, historical facts)

### 3. Generate Script
**IMPORTANT**: Read the full prompt template from `data/PROMPT_TEMPLATE.txt` and follow it precisely.

#### Script Requirements:
- `script.id`: "current"
- `script.topic`: Title (max 60 characters). Use 1-2 UPPERCASE words for gold highlight. Example: "El día que José Smith SANÓ a un pueblo entero"
- `script.background_music`: Choose the BEST fit from "divinity.mp3", "mistery.mp3", "revelations.mp3"
- `script.narration.full_text`: Narration (80-90 seconds, 1:20 minimum)
- `script.narration.visual_assets`: Array of 6-8 visual assets

#### Audio Tags (ElevenLabs V3):
ALLOWED: `[softly]`, `[pause]`, `[pensive]`, `[warmly]`, `[hopeful]`, `[awe]`
FORBIDDEN: `[reverently]` — distorts the voice. NEVER use it.
Max 8 tags per video. Use with emotional intention.

#### Loop Technique:
The video ends with a COMPLETE reflective sentence. The renderer adds 2.5 seconds of music-only tail. When the video restarts, the opening hook connects THEMATICALLY (not literally) with the ending idea.

- NEVER start or end with "..."
- NEVER start or end with a question
- NEVER repeat the first sentence at the end
- The first and last sentences must be DIFFERENT but thematically connected

#### Content Rules:
- Based on REAL experiences of LDS/SUD prophets, leaders, members, and scripture
- Prophets = SUD prophets + biblical prophets + Book of Mormon prophets ONLY
- NO Catholic popes, NO Catholic saints, NO leaders from other denominations
- NO mystery clickbait, NO speculation, NO sensationalism
- At least 2 scripture/doctrinal sources per video

#### Image Prompts:
- Style: LDS sacred art (Greg Olsen, Del Parson, Walter Rane, Simon Dewey)
- Describe scenes as narrative paragraphs, NOT keyword lists
- Specify anatomy, posture, clothing, facial expression, composition, lighting
- Every prompt ends with: "NO CROSSES, NO HALOS, NO WINGS on angels, NO Catholic imagery"
- NO Catholic imagery of any kind (crosses, halos, rosaries, stained glass, cherubs)
- Christ: brown hair, short beard, white/cream robe, red/blue mantle, compassionate expression
- Angels: normal humans in white robes, NO WINGS

#### Validation Checklist (verify before saving):
- [ ] full_text does NOT start with "..."
- [ ] full_text does NOT end with "..."
- [ ] full_text does NOT start with a question
- [ ] full_text does NOT end with a question
- [ ] First and last sentences are NOT the same or similar
- [ ] Tag `[reverently]` does NOT appear anywhere
- [ ] No image_prompt uses "Cinematic oil painting" (must use "Latter-day Saint sacred art")
- [ ] All image_prompts end with "NO CROSSES, NO HALOS, NO WINGS on angels, NO Catholic imagery"
- [ ] full_text has at least 200 words (80-90 seconds)
- [ ] 6-8 visual_assets with consecutive word indices, no gaps
- [ ] No image_prompt says "ANGELS WITHOUT WINGS" (must say "NO WINGS on angels")

Save valid JSON to `data/scripts/current.json`.

### 4. Update Config
Update `data/config.json` with the video metadata:
```json
{
  "video_metadata": {
    "programa": "Ven Sígueme",
    "escritura": "{ESCRITURA}"
  }
}
```

### 5. Generate Images
```bash
python generate_images.py
```
Images are generated using Gemini 2.5 Flash with LDS sacred art styling enforced automatically by `image_generator.py`. The wrapper adds anti-Catholic-imagery and anatomical-correctness constraints to every prompt.

**Retry Strategy**: On 503/capacity errors, waits 30s × attempt number before retry.

### 6. Generate Audio
```bash
python -m src.audio_generator
```
The audio generator includes a **sanitizer** that automatically fixes common script errors before sending to ElevenLabs:
- Strips leading/trailing "..." (prevents garbled syllables)
- Removes `[reverently]` tag (prevents voice distortion)
- Detects and removes duplicate start/end sentences (broken loops)
- Ensures text ends with clean punctuation

### 7. Render Video
```bash
python render_video.py
```
The renderer:
- Applies Ken Burns effect to images
- Renders title with gold (#FFD700) highlight on UPPERCASE words
- Adds navy blue footer bar with program name and scripture reference
- Adds **2.5 seconds of music-only tail** after narration ends for smooth loop transition
- Encodes with NVIDIA NVENC (GPU) or falls back to libx264

### 8. Verify Output
Check `data/output/current.mp4`:
- [ ] Audio syncs with images correctly
- [ ] Title is readable, gold highlight words are visible
- [ ] No garbled audio at start or end
- [ ] 2.5s music tail plays cleanly before loop restart
- [ ] Duration is 1:20+ minimum
- [ ] Images look like LDS sacred art (no Catholic imagery, no deformations)
- [ ] No audio bleeding or distortion

## Content Ideas Generation
To generate 20 video ideas from Ven Sígueme text:
1. Read `data/PROMPT_VIDEO_IDEAS.txt`
2. Append the Ven Sígueme scripture text
3. Generate with Claude — produces table with title, reference, hook, summary, emotion, music
4. Categories: Prophets/Leaders SUD, Doctrine, Testimonies, Christ's Mission

## Output
Final video: `data/output/current.mp4`

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| JSON decode error (UTF-8 BOM) | audio_generator.py uses utf-8-sig encoding |
| Claude CLI outputs extra text | audio_generator.py extracts JSON from mixed content |
| Image generation 503 error | Auto-retry with 30s × attempt backoff |
| Audio tags not working | Text must be 250+ characters for ElevenLabs V3 |
| Garbled syllable at audio end | Sanitizer strips trailing "..." before ElevenLabs |
| `[reverently]` distorts voice | Sanitizer auto-removes it; tag is forbidden |
| Duplicate start/end sentence | Sanitizer detects >70% word overlap and removes duplicate |
| Images look deformed/surreal | image_generator.py enforces anatomical correctness and realism |
| Images have Catholic elements | image_generator.py enforces "NO CROSSES, NO HALOS, NO WINGS" |
| Loop feels like repetition | Use thematic connection, not question-repeat or mid-sentence cut |
| Video too short | Script must be 80-90 seconds (200+ words) |
