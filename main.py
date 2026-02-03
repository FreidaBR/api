from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import base64

app = FastAPI(title="AI Generated Voice Detection API")

# -------------------------
# Flexible Request Schema
# -------------------------
class AudioRequest(BaseModel):
    language: str | None = None
    audio_format: str | None = Field(None, alias="audioFormat")
    audio_base64: str | None = Field(None, alias="audioBase64")

    class Config:
        populate_by_name = True
        extra = "allow"   # allow unknown fields


# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
def predict_voice(
    data: AudioRequest,
    x_api_key: str = Header(...)
):
    # API key check
    if x_api_key != "hackathon123":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Normalize audio format
    audio_format = data.audio_format or data.__dict__.get("audio_format")
    if not audio_format or audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format supported")

    # Normalize base64
    audio_base64 = data.audio_base64 or data.__dict__.get("audio_base64")
    if not audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 missing")

    # Decode base64 safely
    try:
        base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    # Dummy detection (stable)
    return {
        "result": "AI Generated",
        "confidence": 0.85
    }


# -------------------------
# Health Check
# -------------------------
@app.get("/")
def root():
    return {"status": "API is running"}
