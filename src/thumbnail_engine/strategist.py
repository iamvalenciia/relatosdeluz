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


STRATEGY_PROMPT = """Eres un ESTRATEGA EXPERTO en thumbnails de YouTube con 10+ años de experiencia en optimización de CTR.
Tu audiencia: personas de 45-65+ años, hispanohablantes, miembros SUD (Santos de los Últimos Días).
Canal: "Relatos de Luz" - Videos cortos sobre escrituras del programa Ven Sígueme.

REGLA #1 - CLOSE-UP FACE DOMINANTE:
Los thumbnails de alto CTR en YouTube tienen un ROSTRO EN PRIMER PLANO que ocupa 40-60% del frame.
NO cuerpo completo. NO escena panorámica. Un ROSTRO con emoción fuerte es lo que genera clicks.
El rostro debe ser del personaje bíblico/Libro de Mormón principal del video.

REGLA #2 - TEXTO CORTO (2-3 PALABRAS MÁXIMO):
El texto del thumbnail NO es el título del video. Son 2-3 palabras que crean TENSIÓN COGNITIVA.
Una palabra debe ser el gancho emocional en color de acento. Ejemplos:
  "¿POR QUÉ?" (POR QUÉ en acento)
  "La SEÑAL" (SEÑAL en acento)
  "TRAICIÓN Divina" (TRAICIÓN en acento)

REGLA #3 - COMPOSICIÓN DIVIDIDA (SPLIT):
Rostro en un lado (izquierda o derecha), texto en el lado opuesto.
NUNCA texto centrado sobre una ilustración. Siempre separados.

REGLA #4 - FONDO OSCURO/MOODY:
El fondo debe ser OSCURO (negro, azul marino, rojo oscuro) y DESENFOCADO o ABSTRACTO.
El fondo NO debe competir con el rostro. Es un soporte atmosférico, no una escena detallada.

REGLA #5 - ALTO CONTRASTE:
El rostro debe estar bien iluminado contra el fondo oscuro. El contraste atrae el ojo.

REGLA #6 - SISTEMA DE 2 COLORES EN TEXTO:
Blanco como base + UN color de acento (cian, amarillo, rojo, verde).
La palabra de acento es el gancho emocional. NUNCA más de 2 colores en texto.

ANALIZA el video y crea una ESTRATEGIA COMPLETA:

VIDEO:
- Título YouTube: {title}
- Tema: {topic}
- Escritura: {scripture}

GENERA (responde en ESPAÑOL, sé específico y accionable):

1. **ROSTRO** (CRÍTICO - el elemento más importante):
   - ¿Qué personaje bíblico/Libro de Mormón aparece?
   - Expresión facial EXACTA (sorpresa, angustia, determinación, asombro, ira contenida, súplica, etc.)
   - Ángulo de cámara (3/4, frontal ligeramente girado, perfil dramático)
   - Iluminación (lateral dramática, contraluz, luz cenital, Rembrandt lighting)
   - Encuadre: CLOSE-UP de busto/retrato. Desde el pecho hacia arriba. Rostro ocupa 50%+ del frame.
   - Vestimenta visible en la porción de busto (túnica, armadura, manto, etc.)
   - Escala recomendada: 0.7-1.0 relativo al canvas (GRANDE)

2. **TEXTO HOOK** (2-3 palabras MÁXIMO):
   - Las 2-3 palabras que aparecen en el thumbnail.
   - Usa MAYÚSCULAS para la palabra que va en color de acento.
   - El texto debe crear curiosidad SIN repetir el título del video.
   - Debe provocar la pregunta: "¿qué pasó?" o "¿qué significa esto?"

3. **FONDO** (soporte, NO protagonista):
   - Descripción de fondo OSCURO y ABSTRACTO/DESENFOCADO.
   - Puede ser: gradiente de color sólido, humo/niebla, paisaje muy desenfocado, textura abstracta.
   - Color dominante del fondo (oscuro).
   - El fondo NO debe tener detalles que compitan con el rostro.

4. **COMPOSICIÓN** (layout exacto en canvas 1280x720):
   - ¿Rostro a la IZQUIERDA o DERECHA?
   - Posición exacta (x, y) del elemento rostro y su escala.
   - Posición exacta del texto (lado opuesto al rostro).
   - Alineación del texto (left/right) según posición.

5. **PALETA DE COLORES**:
   - Color de texto base: siempre #FFFFFF (blanco).
   - Color de acento para palabra destacada: UN color hex (cian, amarillo, rojo o verde).
   - Color dominante del fondo: hex oscuro.

6. **EMOCIÓN OBJETIVO**: ¿Qué debe sentir el espectador en 0.5 segundos?

7. **CURIOSITY GAP** (2-3 oraciones): ¿Qué pregunta implícita crea el thumbnail + título juntos?"""


ANALYSIS_PROMPT = """Eres un ANALISTA EXPERTO en thumbnails de YouTube con enfoque en CTR (Click-Through Rate).

Analiza este thumbnail para un canal de escrituras SUD llamado "Relatos de Luz".
Audiencia: 45-65+ años, hispanohablantes, miembros SUD.

{title_context}

EVALÚA cada criterio con puntuación individual (1-10):

1. **ROSTRO CLOSE-UP** (peso 30% en CTR): ¿Hay un rostro en primer plano que ocupe 40-60% del frame?
   - Si no hay rostro close-up: máximo 4/10 en CTR total.
   - ¿La expresión es fuerte y legible?
   - ¿El rostro está bien iluminado contra fondo oscuro?
   Puntuación: _/10

2. **TEXTO** (peso 20%): ¿El texto es de 2-3 palabras máximo? ¿Usa sistema de 2 colores (blanco + acento)?
   - Si tiene más de 4 palabras: máximo 5/10.
   - ¿La palabra de acento es el gancho emocional?
   Puntuación: _/10

3. **COMPOSICIÓN SPLIT** (peso 20%): ¿El rostro está en un lado y el texto en el otro?
   - ¿Hay separación clara entre zona de rostro y zona de texto?
   - ¿El layout es limpio y no abarrotado?
   Puntuación: _/10

4. **CONTRASTE Y FONDO** (peso 15%): ¿El fondo es oscuro/moody?
   - ¿El sujeto (rostro) resalta contra el fondo?
   - ¿El fondo NO compite con el sujeto?
   Puntuación: _/10

5. **LEGIBILIDAD MÓVIL** (peso 15%): ¿Se lee todo bien en pantalla de celular?
   - La audiencia 45-65+ usa mucho el celular con texto grande.
   - ¿El texto es lo suficientemente grande?
   - ¿El rostro es reconocible en tamaño pequeño?
   Puntuación: _/10

6. **PREDICCIÓN CTR TOTAL** (1-10): Promedio ponderado de los criterios anteriores.

7. **MEJORAS ESPECÍFICAS** (3-5 acciones concretas con valores exactos):
   Ejemplo: "Mover el rostro 100px a la izquierda", "Reducir texto a 2 palabras",
   "Oscurecer fondo con brightness factor 0.7", "Agregar vignette strength 0.4"

8. **TITLE-THUMBNAIL GAP**: ¿El thumbnail complementa el título sin repetirlo?

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
