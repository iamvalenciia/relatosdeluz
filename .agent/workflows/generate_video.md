---
description: Generate a complete Ven Sígueme video from topic, date, scripture, and context
---

# Video Generation Workflow

This workflow automates the creation of Ven Sígueme videos.

// turbo-all

## Prerequisites
- Claude CLI installed
- ElevenLabs API key configured in `.env`
- Gemini available for image generation

## Steps

### 1. Check for Existing Project
```powershell
python -m src.archive_manager
```
If project exists, choose to archive or continue.

### 2. Collect Video Information
Ask the user to provide:
- **Tema**: Topic title
- **Fecha**: Week date range  
- **Escritura**: Scripture reference
- **Contexto**: Story context and details

### 3. Generate Script with Claude Opus 4
**IMPORTANT**: Use Claude Opus 4 for best quality narration scripts.

Generate the script by creating the JSON directly based on the prompt template structure.
The script must have:
- `script.id`: "current"
- `script.topic`: The topic title
- `script.language`: "es"
- `script.narration.full_text`: Narration with audio tags like [softly], [pause], [warmly], [reverently], [hopeful]
- `script.narration.visual_assets`: Array of 5-7 visual assets with:
  - `visual_asset_id`: "1a", "1b", "1c", etc.
  - `start_word_index` and `end_word_index`: Word ranges (15-25 words per image, don't count audio tags)
  - `image_prompt`: Oil painting style description for Gemini

Save the valid JSON to `data/scripts/current.json`.

**Note**: The audio generator now handles UTF-8 BOM and extracts JSON even if Claude CLI adds extra text.

### 4. Update Config
Update `data/config.json` with the video metadata:
```json
{
  "video_metadata": {
    "programa": "Ven Sígueme",
    "fecha": "{FECHA}",
    "escritura": "{ESCRITURA}"
  }
}
```

### 5. Generate Images with Gemini
For each visual_asset in the script:
1. Get the `image_prompt` from the visual asset
2. Format prompt: `generate a oil painting image with the following description: {prompt}. (family friendly) (SUD/LDS)`
3. Generate 1:1 (square) image
4. Use previous image as reference for visual continuity when possible
5. Save as `data/images/1a.jpeg`, `1b.jpeg`, etc.

**Retry Strategy**: If image generation fails due to capacity (503 error), wait 30 seconds and retry.

### 6. Generate Audio
```powershell
python -m src.audio_generator
```

### 7. Render Video
```powershell
python render_video.py
```

### 8. Verify Output
Check `data/output/current.mp4`:
- Audio syncs with images
- Header covers any logos
- Title text is readable
- No audio bleeding at end

## Output
Final video: `data/output/current.mp4`

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| JSON decode error (UTF-8 BOM) | Fixed in audio_generator.py - uses utf-8-sig encoding |
| Claude CLI outputs extra text | Fixed in audio_generator.py - extracts JSON from mixed content |
| Image generation 503 error | Retry after 30 seconds, Gemini capacity limited |
| Audio tags not working | Ensure text is 250+ characters for ElevenLabs V3 |
