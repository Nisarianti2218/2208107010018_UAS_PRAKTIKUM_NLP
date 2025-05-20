import os
import uuid
import tempfile
import subprocess

# Path ke whisper-cli.exe (pastikan sudah build)
WHISPER_BINARY = r"D:\TUGAS ICA\SEMESTER 6\NLP\2208107010018_UAS PRAK NLP\app\whisper.cpp\build\bin\Release\whisper-cli.exe"

# Ganti ke model SMALL
WHISPER_MODEL_PATH = r"D:\TUGAS ICA\SEMESTER 6\NLP\2208107010018_UAS PRAK NLP\app\whisper.cpp\models\ggml-small.bin"

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    temp_dir = os.path.join(tempfile.gettempdir(), "voice_assistant_stt")
    os.makedirs(temp_dir, exist_ok=True)

    audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
    result_prefix = os.path.join(temp_dir, f"{uuid.uuid4()}")
    result_path = result_prefix + ".txt"

    # Simpan audio sementara
    with open(audio_path, "wb") as f:
        f.write(file_bytes)

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return "[ERROR] Invalid or empty audio file"

    # Cek executable dan model
    if not os.path.isfile(WHISPER_BINARY):
        return f"[ERROR] Whisper executable not found: {WHISPER_BINARY}"
    if not os.path.isfile(WHISPER_MODEL_PATH):
        return f"[ERROR] Whisper model not found: {WHISPER_MODEL_PATH}"

    # Siapkan command whisper
    cmd = [
        WHISPER_BINARY,
        "-m", WHISPER_MODEL_PATH,
        "-f", audio_path,
        "-otxt",
        "-of", result_prefix  # prefix, jangan pakai .txt
    ]

    try:
        print(f"[INFO] Running STT command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[INFO] STT output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Whisper failed with return code {e.returncode}")
        print(f"[ERROR] stderr:\n{e.stderr}")
        return f"[ERROR] Whisper failed: {e.stderr.strip()}"

    # Ambil hasil transkrip
    if not os.path.exists(result_path):
        print(f"[ERROR] Transcription file not found at: {result_path}")
        return "[ERROR] Transcription file not generated"

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
        return transcript if transcript else "[ERROR] Empty transcript generated"
    except Exception as e:
        print(f"[ERROR] Failed to read transcription: {str(e)}")
        return f"[ERROR] Failed to read transcription: {str(e)}"
