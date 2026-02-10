# Helamans Videos - Ven Sigueme Video Generator

Pipeline completa de produccion de videos para el canal **Relatos de Luz**. Genera videos horizontales (16:9, 1920x1080) con narración en español, arte sagrado SUD, y musica de fondo.

## Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Script | Claude + PROMPT_TEMPLATE.txt |
| Imagenes | Gemini 2.5 Flash Image (16:9 oil painting SUD) |
| Audio | ElevenLabs V3 (`eleven_v3`) + Whisper timestamps |
| Video | PyAV + Pillow (Ken Burns, lower third, NVENC GPU) |
| Thumbnail | Pillow (gradient, gold highlights, badges) |
| Integracion | MCP Server para Claude Desktop |

---

## Requisitos

```bash
pip install -r requirements.txt
```

### Variables de entorno (`.env`)

```env
ELEVEN_LABS_API_KEY=tu_api_key_de_elevenlabs
GEMINI_API_KEY=tu_api_key_de_gemini
```

### Fuentes (Google Fonts → `data/fonts/`)

- `Montserrat-ExtraBold.ttf`
- `Montserrat-Bold.ttf`
- `Montserrat-SemiBold.ttf`
- `Montserrat-Medium.ttf`

---

## Configuracion MCP para Claude Desktop

### 1. Instalar dependencia MCP

```bash
pip install mcp
```

### 2. Configurar Claude Desktop

Abre el archivo de configuracion de Claude Desktop:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`

Agrega el servidor MCP:

```json
{
  "mcpServers": {
    "helamans-videos": {
      "command": "python",
      "args": ["C:/Users/juanf/OneDrive/Escritorio/helamans-videos/mcp_server.py"],
      "env": {
        "ELEVEN_LABS_API_KEY": "tu_key",
        "GEMINI_API_KEY": "tu_key"
      }
    }
  }
}
```

### 3. Reiniciar Claude Desktop

Cierra y abre Claude Desktop. Veras las herramientas disponibles en el icono de martillo.

---

## Workflow Completo (Paso a Paso)

### Paso 0: Verificar Estado del Proyecto

> **MCP Tool**: `check_project_status`

Esto te dice en que fase esta el workspace (vacio, script listo, imagenes listas, audio listo, renderizado, completo) y que archivos existen.

Si hay un proyecto anterior:

> **MCP Tool**: `archive_project` con `confirm: true`

Esto archiva todo a `data/archive/{timestamp}/` y limpia el workspace.

---

### Paso 1: Generar el Script

> **MCP Tool**: `get_prompt_template` con `tema`, `escritura`, `contexto`

Esto devuelve el prompt completo de `PROMPT_TEMPLATE.txt` con los valores rellenados. **Usa este prompt como instruccion para que Claude genere el script JSON**.

#### Prompt de ejemplo para Claude:

```
Usa la herramienta get_prompt_template con:
- tema: "La Noche que Jose Smith Recibio el Sacerdocio de Melquisedec"
- escritura: "DyC 84:19-22; JS-Historia 1:68-72"
- contexto: "En mayo de 1829, Jose Smith y Oliver Cowdery oraron a orillas
  del rio Susquehanna. Pedro, Santiago y Juan aparecieron como seres
  resucitados y les confirieron el Sacerdocio de Melquisedec."

Ahora genera el script JSON siguiendo el prompt template al pie de la letra.
```

#### Reglas criticas del script:

**Narración (full_text)**:
- 350-450 palabras (2-3 minutos de duracion)
- NO empezar ni terminar con `...`
- NO empezar ni terminar con una pregunta
- Primera y ultima oracion diferentes (no repetir para "loop")
- Minimo 2 fuentes doctrinales/escriturales

**Audio Tags** (ElevenLabs V3 — para enganche y retencion):
```
PERMITIDOS:  [softly]  [pause]  [pensive]  [warmly]  [hopeful]  [awe]
PROHIBIDO:   [reverently]  ← distorsiona la voz, NUNCA usar
```

**Uso estrategico de audio tags para maxima retencion**:
- `[pause]` → Silencio dramatico ANTES de una verdad importante. Da peso y gravedad.
- `[softly]` → Momentos intimos, sagrados. Como bajar la voz para un secreto.
- `[warmly]` → Conexion directa con el espectador. "Esto es para ti."
- `[hopeful]` → Climax positivo, la promesa cumplida.
- `[awe]` → Momentos de asombro ante la grandeza de Dios.
- `[pensive]` → Reflexion profunda, cuando el narrador procesa algo conmovedor.
- Maximo 12 tags por video. No poner dos tags seguidos.
- Alternar tags — no repetir el mismo mas de 3 veces.
- El texto DEBE tener 250+ caracteres para que los tags se activen en ElevenLabs V3.

**Visual Assets (10-14 imagenes)**:
- `visual_asset_id`: "1a", "1b", "1c", etc.
- `start_word_index` / `end_word_index`: consecutivos sin saltos, empezando en 0
- `image_prompt`: parrafo narrativo (NO keywords), estilo "Latter-day Saint sacred art"
- Cada prompt termina con: `"NO CROSSES, NO HALOS, NO WINGS on angels, NO Catholic imagery"`

**Musica de fondo**:
- `divinity.mp3` → Amor divino, familia eterna, sacrificio
- `mistery.mp3` → Visiones, revelaciones, asombro sagrado
- `revelations.mp3` → Sacerdocio, Segunda Venida, victoria de fe

**Metadata**:
- `title_youtube`: Max 70 chars, optimizado para CTR
- `title_internal`: Max 80 chars, descriptivo para dentro del video
- `description`: 150-200 palabras, SEO
- `tags`: 10-15 tags relevantes
- `hashtags`: incluir `#RelatosDeLuz` y `#VenSigueme`

---

### Paso 2: Guardar y Validar el Script

> **MCP Tool**: `save_script` con `script_json` y `escritura`

Valida automaticamente el JSON contra todas las reglas. Si hay errores, corrige y vuelve a guardar.

Para validar un script ya guardado:

> **MCP Tool**: `validate_script`

---

### Paso 3: Generar Imagenes

> **MCP Tool**: `generate_images`

Genera todas las imagenes con Gemini 2.5 Flash Image en formato 16:9. El wrapper automaticamente:
- Agrega estilo "Latter-day Saint sacred art" (Greg Olsen, Del Parson, Walter Rane)
- Agrega restricciones anti-catolicismo (NO cruces, NO halos, NO alas)
- Agrega requisitos anatomicos (proporciones correctas, expresiones realistas)
- Reintenta en errores 503 (30s x intento)
- Salta imagenes que ya existen (usa `force: true` para regenerar)

---

### Paso 4: Generar Audio

> **MCP Tool**: `generate_audio`

Proceso automatico:
1. **Sanitizador** limpia el texto (quita `...`, remueve `[reverently]`, detecta duplicados)
2. **ElevenLabs V3** genera audio MP3 192kbps con la voz configurada
3. **Whisper** genera timestamps word-level para sincronizacion imagen-audio

---

### Paso 4.5: Alinear Timestamps (CRITICO)

> **MCP Tool**: `align_timestamps` con `auto_fix: false` (dry-run) o `auto_fix: true` (guardar)

**Por que es necesario**: Los `word_index` del script se calculan sobre el texto original, pero:
- El sanitizador puede modificar el texto (quitar tags, ellipsis, duplicados)
- Whisper tokeniza las palabras del audio de forma distinta (puntuacion, contracciones)
- El resultado es que `visual_assets[i].end_word_index` NO coincide con el indice real de Whisper

**Que hace esta herramienta**:
1. Tokeniza el texto del script (sin audio tags) → "palabras del script"
2. Compara fuzzy-match con las palabras de Whisper (por similitud de texto)
3. Construye un mapeo: `script_word_index → whisper_word_index`
4. Remapea todos los `start_word_index` / `end_word_index` de `visual_assets`
5. Verifica continuidad (sin gaps entre assets)
6. Genera reporte detallado con: matches exactos, fuzzy, faltantes, cambios

**Workflow recomendado**:
```
1. Primero dry-run para revisar:
   align_timestamps con auto_fix: false
   → Revisar el reporte, verificar que los remapeos tienen sentido

2. Si todo se ve bien, aplicar:
   align_timestamps con auto_fix: true
   → Guarda el script corregido automaticamente
```

**Tambien se puede ejecutar por CLI**:
```bash
# Dry-run (solo reporte)
python -m src.timestamp_aligner

# Auto-fix (guardar cambios)
python -m src.timestamp_aligner --fix
```

---

### Paso 5: Renderizar Video

> **MCP Tool**: `render_video`

Renderiza video final 1920x1080 H.264 MP4:
- Ken Burns effect (zoom 1.0→1.08 con pan suave)
- Lower third profesional estilo noticiero (badge "VEN SIGUEME" + titulo + escritura)
- Narración boosted 1.15x + musica dinamica (15% durante voz, 40% despues, fade out)
- Cola de 5 segundos de musica para loop suave
- GPU: NVIDIA NVENC (RTX) o fallback CPU libx264

---

### Paso 6: Generar Metadata para YouTube

> **MCP Tool**: `get_metadata`

Devuelve titulo, descripcion, tags y hashtags listos para YouTube Studio.

---

### Paso 7: Generar Thumbnail

#### Prompt para que Claude elija la mejor imagen:

```
Revisa las imagenes generadas en data/images/. Selecciona la mas dramatica
y visualmente impactante para la miniatura. Considera:
- Que tenga un personaje o escena central clara
- Buena iluminacion dramatica
- Que se vea bien en tamaño pequeño (mobile)

Luego genera la miniatura con un titulo corto de 3-5 palabras.
Usa MAYUSCULAS en 1-2 palabras clave para que aparezcan en dorado/rojo.
```

> **MCP Tool**: `generate_thumbnail` con:
> - `asset_id`: el ID de la imagen elegida (e.g. "1c")
> - `custom_title`: titulo corto con MAYUSCULAS (e.g. "Los Dias de NOE\nYa LLEGARON")
> - `zoom`: 1.0-1.3
> - `pan_x` / `pan_y`: -0.5 a 0.5

Para ajustar la configuracion avanzada (colores, badges, gradient):

> **MCP Tool**: `configure_thumbnail` con `config_updates`

---

### Paso 8: Verificar y Publicar

Checklist final:
- [ ] Video (`data/output/current.mp4`): audio sincronizado, titulo legible, sin distorsion
- [ ] Thumbnail (`data/output/thumbnail.png`): dramatica, texto legible en mobile, badges visibles
- [ ] Metadata (`data/output/metadata.txt`): titulo, descripcion, tags, hashtags

---

## Generar Ideas de Video (Opcional)

> **MCP Tool**: `get_video_ideas_prompt` con `ven_sigueme_text`

Pega el texto de la leccion semanal de Ven Sigueme y obtendras un prompt que genera 20 ideas de video organizadas por categoria.

---

## Estructura del Proyecto

```
helamans-videos/
├── mcp_server.py              ← Servidor MCP para Claude Desktop
├── render_video.py            ← Wrapper para renderizar video
├── generate_images.py         ← Wrapper para generar imagenes
├── generate_thumbnail.py      ← Wrapper para generar thumbnail
├── requirements.txt           ← Dependencias Python
├── src/
│   ├── audio_generator.py     ← ElevenLabs V3 + Whisper
│   ├── video_renderer.py      ← PyAV + Ken Burns + lower third
│   ├── image_generator.py     ← Gemini 2.5 Flash Image
│   ├── thumbnail_generator.py ← YouTube thumbnail
│   └── archive_manager.py     ← Archivado de proyectos
├── data/
│   ├── config.json            ← Configuracion principal
│   ├── thumbnail_config.json  ← Configuracion de thumbnails
│   ├── PROMPT_TEMPLATE.txt    ← Prompt maestro para scripts
│   ├── PROMPT_VIDEO_IDEAS.txt ← Prompt para generar ideas
│   ├── scripts/
│   │   └── current.json       ← Script activo
│   ├── images/                ← Imagenes generadas (1920x1080 PNG)
│   ├── audio/
│   │   ├── current.mp3        ← Audio narración
│   │   └── current_timestamps.json ← Timestamps word-level
│   ├── music/                 ← Musica de fondo
│   ├── fonts/                 ← Fuentes Montserrat
│   ├── output/
│   │   ├── current.mp4        ← Video final
│   │   ├── thumbnail.png      ← Thumbnail YouTube
│   │   └── metadata.txt       ← Metadata para YouTube
│   └── archive/               ← Proyectos archivados
└── .agent/
    └── workflows/
        └── generate_video.md  ← Workflow para Claude Code CLI
```

---

## MCP Tools Reference

| Tool | Descripcion | Cuando Usar |
|---|---|---|
| `check_project_status` | Estado actual del workspace | Al inicio de cada sesion |
| `archive_project` | Archivar y limpiar workspace | Antes de nuevo video |
| `get_prompt_template` | Obtener prompt para generar script | Paso 1 |
| `save_script` | Guardar y validar script JSON | Paso 2 |
| `validate_script` | Validar script existente | Verificacion extra |
| `generate_images` | Generar imagenes con Gemini | Paso 3 |
| `generate_audio` | Generar audio + timestamps | Paso 4 |
| `align_timestamps` | Alinear word indices script↔Whisper | Paso 4.5 (CRITICO) |
| `render_video` | Renderizar video final | Paso 5 |
| `get_metadata` | Obtener metadata para YouTube | Paso 6 |
| `generate_thumbnail` | Generar thumbnail con overrides | Paso 7 |
| `configure_thumbnail` | Ajustar config de thumbnail | Paso 7 (avanzado) |
| `get_video_ideas_prompt` | Prompt para 20 ideas de video | Ideacion |
| `list_archives` | Listar proyectos archivados | Consulta |

---

## Ejemplo de Sesion Completa en Claude Desktop

```
Usuario: Quiero hacer un video sobre la Primera Vision de Jose Smith
         Escritura: JS-Historia 1:15-20
         Contexto: Jose Smith tenia 14 años en la primavera de 1820.
                   Fue al bosque a orar despues de leer Santiago 1:5.
                   Vio a Dios el Padre y a Jesucristo.

Claude:  [usa check_project_status → workspace vacio]
         [usa get_prompt_template con tema, escritura, contexto]
         [genera script JSON completo siguiendo el template]
         [usa save_script → validacion exitosa]
         [usa generate_images → 12/12 imagenes generadas]
         [usa generate_audio → 2:15 de audio, 387 palabras]
         [usa align_timestamps auto_fix=false → reporte muestra 3 indices desalineados]
         [usa align_timestamps auto_fix=true → script corregido y guardado]
         [usa render_video → video de 2:20 generado]
         [usa get_metadata → titulo, descripcion, tags listos]
         [revisa imagenes, elige la mas dramatica]
         [usa generate_thumbnail con asset_id="1c",
          custom_title="La VISION que\nCambio el MUNDO"]

         ¡Listo! Tu video esta en data/output/current.mp4
         Thumbnail en data/output/thumbnail.png
```

---

## Uso sin MCP (CLI directo)

Si prefieres ejecutar los scripts directamente:

```bash
# 1. Archivar proyecto anterior
python -m src.archive_manager

# 2. Guardar script en data/scripts/current.json (manual)

# 3. Generar imagenes
python generate_images.py

# 4. Generar audio
python -m src.audio_generator

# 4.5. Alinear timestamps (CRITICO - antes de renderizar)
python -m src.timestamp_aligner        # dry-run
python -m src.timestamp_aligner --fix  # aplicar cambios

# 5. Renderizar video
python render_video.py

# 6. Generar thumbnail
python generate_thumbnail.py
```

---

## Problemas Comunes

| Problema | Solucion |
|---|---|
| Imagenes no cambian a tiempo | Correr `align_timestamps --fix` para realinear indices |
| Imagen aparece en momento equivocado | Los word_index del script no coinciden con Whisper. Usar `align_timestamps` |
| Audio distorsionado | Verificar que no haya `[reverently]` en el script |
| Silaba cortada al inicio/final | El sanitizador quita `...` automaticamente |
| Tags de audio no funcionan | Texto debe tener 250+ caracteres |
| Error 503 en imagenes | Reintento automatico con backoff (30s x intento) |
| Imagenes con cruces/halos | image_generator.py agrega restricciones automaticamente |
| Video muy corto | Script debe tener 350-450 palabras (2-3 min) |
| JSON con BOM | audio_generator usa encoding `utf-8-sig` |
| Fuentes no encontradas | Descargar Montserrat de Google Fonts → `data/fonts/` |
| GPU no detectada | Fallback automatico a CPU (libx264) |
