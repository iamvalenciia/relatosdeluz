---
description: Generate a complete Ven Sígueme video from topic, date, scripture, and context
---

# Video Generation Workflow

This workflow automates the creation of Ven Sígueme videos with storytelling-driven content.

// turbo-all

## Prerequisites
- Claude CLI installed
- ElevenLabs API key configured in `.env`
- Gemini available for image generation

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
- **Contexto**: Story context and details (real experiences, scripture references, historical facts)

### 3. Generate Script with Claude Opus 4
**IMPORTANT**: Use Claude Opus 4 for best quality narration scripts.

Read the prompt template from `data/PROMPT_TEMPLATE.txt` and follow it precisely.

The script must have:
- `script.id`: "current"
- `script.topic`: The topic title (max 60 characters, use 1-2 UPPERCASE words for emphasis)
- `script.background_music`: One of "divinity.mp3", "mistery.mp3", "revelations.mp3"
- `script.narration.full_text`: Narration with audio tags (80-90 seconds, 1:20 minimum)
  - Tags: [softly], [pause], [warmly], [reverently], [hopeful], [pensive], [awe]
  - Max 8 tags per video
  - Must use TRUE NARRATIVE LOOP (not question-repeat)
  - Must be based on real experiences, scriptures, and official Church sources
- `script.narration.visual_assets`: Array of 6-8 visual assets with:
  - `visual_asset_id`: "1a", "1b", "1c", etc.
  - `start_word_index` and `end_word_index`: Word ranges (12-18 words per image, don't count audio tags)
  - `image_prompt`: Detailed narrative oil painting description (Old Masters style, no keywords)

Save the valid JSON to `data/scripts/current.json`.

**Loop technique**: The video ENDS with an incomplete sentence "..." and BEGINS with "..." completing it. When the video loops, the sentence flows naturally as one. NEVER use question-to-question loops.

**Title highlight**: Words in UPPERCASE in the topic will be rendered in gold (#FFD700) in the video. Use 1-2 uppercase words for emphasis. Example: "Lo que Cristo hizo ANTES de nacer en Belén"

**Note**: The audio generator handles UTF-8 BOM and extracts JSON even if Claude CLI adds extra text.

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

### 5. Generate Images with Gemini
```bash
python generate_images.py
```

For each visual_asset in the script, images are generated using Gemini 2.5 Flash with enhanced prompting:
- Old Masters oil painting style (Rembrandt, Caravaggio)
- Anatomically correct proportions
- Naturalistic facial features
- No surrealism, no deformations, no fantasy elements
- Family friendly, Latter-day Saint reverent atmosphere

**Retry Strategy**: If image generation fails due to capacity (503 error), wait 30 seconds and retry.

### 6. Generate Audio
```bash
python -m src.audio_generator
```

### 7. Render Video
```bash
python render_video.py
```

### 8. Verify Output
Check `data/output/current.mp4`:
- Audio syncs with images
- Title text is readable (gold highlight words visible)
- Loop transition feels seamless (end flows into beginning)
- Duration is 1:20+ minimum
- No audio bleeding at end

## Content Ideas Generation
To generate 20 video ideas from Ven Sígueme text:
1. Read the prompt template from `data/PROMPT_VIDEO_IDEAS.txt`
2. Append the Ven Sígueme text
3. Generate with Claude Opus 4

## Output
Final video: `data/output/current.mp4`

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| JSON decode error (UTF-8 BOM) | Fixed in audio_generator.py - uses utf-8-sig encoding |
| Claude CLI outputs extra text | Fixed in audio_generator.py - extracts JSON from mixed content |
| Image generation 503 error | Retry after 30 seconds, Gemini capacity limited |
| Audio tags not working | Ensure text is 250+ characters for ElevenLabs V3 |
| Images look deformed/surreal | Enhanced prompt in image_generator.py enforces realism |
| Loop feels like repetition | Use narrative sentence loop, not question loop |
| Video too short | Script must be 80-90 seconds minimum (1:20+) |
