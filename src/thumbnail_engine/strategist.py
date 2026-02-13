"""
Thumbnail Strategist - AI-powered strategy and analysis using Gemini.

Uses Gemini 2.5 Flash for:
  - Strategy generation: complete thumbnail creative plan
  - Analysis: CTR prediction, strengths, weaknesses, improvements
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def _get_client():
    """Get Gemini client."""
    from google import genai
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env")
    return genai.Client(api_key=GEMINI_API_KEY)


STRATEGY_PROMPT = """Eres un ESTRATEGA EXPERTO en thumbnails de YouTube con 10+ años de experiencia.
Tu audiencia: personas de 45-65+ años, hispanohablantes, miembros SUD (Santos de los Últimos Días).
Canal: "Relatos de Luz" - Videos cortos sobre escrituras del programa Ven Sígueme.

ANALIZA el video y crea una ESTRATEGIA COMPLETA de thumbnail:

VIDEO:
- Título YouTube: {title}
- Tema: {topic}
- Escritura: {scripture}

GENERA:
1. **CURIOSITY GAP** (3-4 oraciones): ¿Qué pregunta debe crear el thumbnail que el título no responda completamente? ¿Qué tensión cognitiva genera el click?

2. **TEXTO HOOK** (2-4 palabras): El texto que aparecerá EN el thumbnail. Debe ser provocativo, crear curiosidad, NO repetir el título. Usa MAYÚSCULAS para las palabras que deben destacar en color de acento. Ejemplos: "¿Y AHORA QUÉ?", "La SEÑAL Olvidada", "ARCO de GUERRA"

3. **FONDO** (descripción detallada): Escena de fondo 16:9. Describe la atmósfera, iluminación, colores, ambiente. Debe ser dramático y visualmente impactante.

4. **ELEMENTOS** (1-2 elementos): Personajes u objetos que necesitan generarse por separado para componer sobre el fondo. Describe pose, expresión, vestimenta, ángulo.

5. **COMPOSICIÓN**: Dónde va cada elemento en el canvas 1280x720:
   - Posición del fondo
   - Posición y escala de cada elemento
   - Posición del texto
   - Efectos visuales recomendados (vignette, gradient, glow)

6. **PALETA DE COLORES**: 3-4 colores hex que dominan el thumbnail.

7. **EMOCIÓN OBJETIVO**: ¿Qué debe sentir el espectador en los primeros 0.5 segundos?

Responde en ESPAÑOL. Sé específico y accionable."""


ANALYSIS_PROMPT = """Eres un ANALISTA EXPERTO en thumbnails de YouTube con enfoque en CTR (Click-Through Rate).

Analiza este thumbnail para un canal de escrituras SUD llamado "Relatos de Luz".
Audiencia: 45-65+ años, hispanohablantes, miembros SUD.

{title_context}

EVALÚA:

1. **PREDICCIÓN CTR** (1-10): Estimación de efectividad. 1-3 = bajo, 4-6 = promedio, 7-8 = bueno, 9-10 = excelente.

2. **FORTALEZAS** (3-5 puntos): ¿Qué funciona bien? Contraste, composición, texto, colores, emoción.

3. **DEBILIDADES** (3-5 puntos): ¿Qué puede mejorar? Legibilidad, composición, impacto visual, curiosidad.

4. **MEJORAS ESPECÍFICAS** (3-5 acciones concretas): Instrucciones exactas para mejorar.
   Ejemplo: "Mover el personaje 100px a la derecha", "Aumentar el tamaño del texto a 84px",
   "Agregar vignette con strength 0.3", "Cambiar color del texto a #FFE500".

5. **TITLE-THUMBNAIL GAP**: ¿El thumbnail complementa el título sin repetirlo? ¿Genera curiosidad adicional?

6. **LEGIBILIDAD MÓVIL**: ¿Se lee bien en pantalla pequeña de celular? (La audiencia 45-65+ usa mucho el celular)

Responde en ESPAÑOL. Sé específico y accionable."""


def generate_strategy(
    video_title: str,
    video_topic: str,
    scripture: str = "",
    mood: str = "",
) -> str:
    """
    Generate a complete thumbnail strategy using Gemini text model.
    Returns formatted strategy text.
    """
    from google.genai import types

    client = _get_client()
    prompt = STRATEGY_PROMPT.format(
        title=video_title,
        topic=video_topic,
        scripture=scripture or "No especificada",
    )
    if mood:
        prompt += f"\n\nMood deseado: {mood}"

    print("  Generating thumbnail strategy...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )

    result = response.text or "ERROR: No response from Gemini"
    print(f"  Strategy generated ({len(result)} chars)")
    return result


def analyze_thumbnail(
    image_path: Path,
    video_title: str = "",
) -> str:
    """
    Analyze a generated thumbnail using Gemini multimodal (vision + text).
    Returns CTR analysis with specific improvements.
    """
    from google.genai import types

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = _get_client()
    img = Image.open(image_path)

    title_context = ""
    if video_title:
        title_context = f"Título del video: \"{video_title}\""

    prompt = ANALYSIS_PROMPT.format(title_context=title_context)

    print(f"  Analyzing thumbnail: {image_path.name}")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img],
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )

    result = response.text or "ERROR: No response from Gemini"
    print(f"  Analysis complete ({len(result)} chars)")
    return result
