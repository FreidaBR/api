from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64

app = FastAPI(title="AI Generated Voice Detection API")

# -------------------------
# Request Body Schema
# -------------------------
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
def predict_voice(
    data: AudioRequest,
    x_api_key: str = Header(...)
):
    # 1️⃣ Validate API key
    if x_api_key != "hackathon123":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2️⃣ Validate audio format
    if data.audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format supported")

    # 3️⃣ Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    # (Optional) You could save or process audio here
    # with open("temp.mp3", "wb") as f:
    #     f.write(audio_bytes)

    # 4️⃣ Dummy detection logic (hackathon-safe)
    # Replace with ML later
    result = "AI Generated"
    confidence = 0.85

    # 5️⃣ Return response
    return {
        "result": result,
        "confidence": confidence
    }


# -------------------------
# Health Check
# -------------------------
@app.get("/")
def root():
    return {"status": "API is running"}
