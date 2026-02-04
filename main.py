"""
AI-Generated Voice Detection API (Multi-Language)
Accepts Base64-encoded MP3 audio and returns classification (AI_GENERATED / HUMAN),
confidence score, and forensic explanation. Supports: Tamil, English, Hindi, Malayalam, Telugu.
"""

import base64
import os
from enum import Enum
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# Optional: use google-genai for Gemini (set GEMINI_API_KEY or GOOGLE_API_KEY in env)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
except ImportError:
    GEMINI_AVAILABLE = False


# --- Supported languages (match problem statement) ---
class SupportedLanguage(str, Enum):
    Tamil = "Tamil"
    English = "English"
    Hindi = "Hindi"
    Malayalam = "Malayalam"
    Telugu = "Telugu"


# --- Request / Response schemas ---
class VoiceDetectionRequest(BaseModel):
    language: SupportedLanguage
    audioFormat: str = Field(..., description="Must be 'mp3'")
    audioBase64: str = Field(..., description="Base64-encoded MP3 audio")


class VoiceDetectionSuccessResponse(BaseModel):
    status: Literal["success"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class VoiceDetectionErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str


# --- Gemini forensic prompt (tuned for high accuracy) ---
SYSTEM_INSTRUCTION = (
    "You are a world-class audio forensic analyst. Your task is to determine whether "
    "a voice sample is AI-generated (synthetic/TTS/deepfake) or human-generated. "
    "Your analysis must be extremely critical: if you detect ANY signs of neural synthesis "
    "(vocoder artifacts, unnatural phase coherence, metallic timbre, artificial breath patterns), "
    "classify as AI_GENERATED. Be precise with your confidence score (0.0 to 1.0) based on "
    "the strength and number of artifacts found. Prefer slight bias toward AI when uncertain "
    "to reduce false negatives in security-sensitive use cases."
)

def _build_forensic_prompt(language: str) -> str:
    return f"""Perform a deep forensic acoustic analysis on the provided audio sample to detect if it is AI-generated (Deepfake/TTS/neural synthesis) or Human.

Target Language: {language}

Analysis framework (evaluate each):
1. **Spectral artifacts**: Listen for high-frequency metallic buzzing, phasing, or "vocoder" quality common in neural vocoders.
2. **Breath dynamics**: Humans have natural, irregular micro-breaths between phrases. AI often has no breaths, or repetitive/inserted breath sounds that don't match speech exertion.
3. **Prosody & intonation**: Check for unnatural flatness in pitch or overly perfect rhythm. Humans vary speed and pitch with emotion and emphasis.
4. **Noise floor**: Humans typically have a consistent background noise floor. AI may show "digital silence" (near-zero amplitude) between words or spectral gating artifacts.
5. **Glottal artifacts**: Vocal fry and glottal stops. AI often fails to replicate the chaotic nature of human vocal folds.
6. **Consonant clarity**: Overly crisp or uniform consonants can indicate TTS; humans have more variation.

Weigh the evidence and output your classification, confidence (0.0–1.0), and a concise technical explanation citing specific observations (e.g., "Metallic phasing at ~8 kHz", "No natural breath pauses")."""


def analyze_audio_with_gemini(audio_base64: str, language: str) -> dict:
    """Call Gemini to classify the audio. Returns dict with classification, confidenceScore, explanation."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set; cannot run detection.")

    # Strip data URI if present
    clean_b64 = audio_base64.split(",", 1)[-1].strip()
    try:
        audio_bytes = base64.b64decode(clean_b64, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 audio: {e}") from e

    client = genai.Client()
    prompt = _build_forensic_prompt(language.value)

    # Use a model with strong reasoning; prefer 2.5 Pro for accuracy (or 2.5 Flash for speed)
    model_id = os.environ.get("VOICE_DETECTION_MODEL", "gemini-2.5-pro")
    # Fallback if 2.5-pro not available in your project
    # model_id = "gemini-2.5-flash"

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "classification": {"type": "STRING", "enum": ["AI_GENERATED", "HUMAN"]},
                "confidenceScore": {"type": "NUMBER", "description": "Float 0.0 to 1.0; 1.0 = absolute certainty."},
                "explanation": {"type": "STRING", "description": "Technical forensic explanation citing specific artifacts."},
            },
            "required": ["classification", "confidenceScore", "explanation"],
        },
        max_output_tokens=2048,
        temperature=0.2,  # Lower temperature for more consistent, conservative classifications
    )

    # Optional: enable thinking for complex reasoning (if model supports it, e.g. gemini-2.5-pro)
    try:
        config.thinking_config = types.ThinkingConfig(thinking_budget=1024)
    except Exception:
        pass

    contents = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
    ]

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config,
    )

    text = response.text
    if not text or not text.strip():
        raise RuntimeError("Empty response from Gemini.")

    import json
    result = json.loads(text.strip())
    # Normalize key to match API (camelCase)
    if "confidence_score" in result:
        result["confidenceScore"] = result.pop("confidence_score")
    return result


# --- API key validation (use env in production) ---
VALID_API_KEYS: set[str] = set(
    filter(None, (os.environ.get("VOICE_DETECTION_API_KEYS", "sk_test_123456789").replace(" ", "").split(",")))
)
if not VALID_API_KEYS:
    VALID_API_KEYS.add("sk_test_123456789")


app = FastAPI(
    title="AI-Generated Voice Detection API",
    description="Determines whether a voice sample is AI-generated or human. Supports Tamil, English, Hindi, Malayalam, Telugu.",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Root route so the base URL doesn't return 404."""
    return {
        "message": "AI-Generated Voice Detection API",
        "docs": "/docs",
        "health": "/health",
        "detection": "POST /api/voice-detection (header: x-api-key)",
    }


@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionSuccessResponse | VoiceDetectionErrorResponse,
)
async def voice_detection(
    body: VoiceDetectionRequest,
    x_api_key: str | None = Header(None, alias="x-api-key"),
):
    # 1. Validate API key
    api_key = (x_api_key or "").strip()
    if not api_key or api_key not in VALID_API_KEYS:
        return VoiceDetectionErrorResponse(
            status="error",
            message="Invalid API key or malformed request",
        )

    # 2. Validate required fields (Pydantic already does this; extra checks for clarity)
    if not body.language or not body.audioFormat or not body.audioBase64:
        return VoiceDetectionErrorResponse(
            status="error",
            message="Missing required fields: language, audioFormat, or audioBase64",
        )

    # 3. Validate language
    if body.language not in SupportedLanguage:
        return VoiceDetectionErrorResponse(
            status="error",
            message=f"Unsupported language. Supported: {', '.join(e.value for e in SupportedLanguage)}",
        )

    # 4. Validate audio format
    if body.audioFormat.strip().lower() != "mp3":
        return VoiceDetectionErrorResponse(
            status="error",
            message='Invalid audioFormat. Only "mp3" is supported.',
        )

    # 5. Run detection
    try:
        analysis = analyze_audio_with_gemini(body.audioBase64, body.language)
        classification = analysis.get("classification") or "HUMAN"
        if classification not in ("AI_GENERATED", "HUMAN"):
            classification = "HUMAN"
        confidence = float(analysis.get("confidenceScore", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        explanation = str(analysis.get("explanation", "No explanation provided.")).strip() or "No explanation provided."

        return VoiceDetectionSuccessResponse(
            status="success",
            language=body.language.value,
            classification=classification,
            confidenceScore=round(confidence, 4),
            explanation=explanation,
        )
    except ValueError as e:
        return VoiceDetectionErrorResponse(status="error", message=str(e))
    except Exception as e:
        # Log in production; return generic message to client
        return VoiceDetectionErrorResponse(
            status="error",
            message="Internal processing error during voice analysis.",
        )


@app.get("/health")
async def health():
    return {"status": "ok", "gemini_configured": GEMINI_AVAILABLE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
