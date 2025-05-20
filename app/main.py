from dotenv import load_dotenv
import os

# Load environment variables dari .env
load_dotenv()
print("GEMINI_API_KEY =", os.getenv("GEMINI_API_KEY"))  # cek apakah key terbaca

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech
import base64
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("voice-assistant")

app = FastAPI(title="Voice AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # batasi di produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Voice AI Assistant API is running"}

@app.post("/voice-chat")
async def voice_chat(request: Request, file: UploadFile = File(...)):
    logger.info(f"Received request from {request.client.host} with file: {file.filename}")
    logger.info(f"Request headers: {request.headers}")

    contents = await file.read()
    file_size = len(contents)
    logger.info(f"File received: {file_size} bytes")

    if not contents or file_size == 0:
        logger.error("Empty file received")
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    logger.info("Starting speech-to-text processing")
    transcript = transcribe_speech_to_text(contents, file_ext=os.path.splitext(file.filename)[1])
    if transcript.startswith("[ERROR]"):
        logger.error(f"STT error: {transcript}")
        return JSONResponse(status_code=500, content={"error": transcript})
    logger.info(f"Transcribed text: {transcript}")

    logger.info("Generating LLM response")
    response_text = generate_response(transcript)
    if response_text.startswith("[ERROR]"):
        logger.error(f"LLM error: {response_text}")
        return JSONResponse(status_code=500, content={"error": response_text})
    logger.info(f"LLM Response: {response_text}")

    logger.info("Starting text-to-speech processing")
    audio_path = transcribe_text_to_speech(response_text)
    if not audio_path or audio_path.startswith("[ERROR]"):
        logger.error(f"TTS error: {audio_path}")
        return JSONResponse(status_code=500, content={"error": f"Failed to generate speech: {audio_path}"})

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found at: {audio_path}")
        return JSONResponse(status_code=500, content={"error": f"Audio file not found at: {audio_path}"})

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return JSONResponse(
        content={
            "response": response_text,
            "audio_base64": audio_b64,
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)