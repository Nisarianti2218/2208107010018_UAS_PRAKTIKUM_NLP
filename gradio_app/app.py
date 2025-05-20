import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile
import base64

def voice_chat(audio):
    if audio is None:
        return None, "Tidak ada input audio."

    sr, audio_data = audio

    # Simpan sebagai file .wav sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    # Kirim ke backend FastAPI
    with open(audio_path, "rb") as f:
        files = {"file": ("voice.wav", f, "audio/wav")}
        response = requests.post("http://localhost:8000/voice-chat", files=files)

    if response.status_code == 200:
        result = response.json()

        # Decode base64 audio
        audio_b64 = result.get("audio_base64")
        response_text = result.get("response", "[Tanpa teks]")

        output_audio_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        with open(output_audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        return output_audio_path, response_text
    else:
        return None, f"Terjadi error dari server: {response.status_code}"

# UI Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Voice Chatbot")
    gr.Markdown("Berbicara langsung ke mikrofon dan dapatkan jawaban suara dari asisten AI.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources="microphone", type="numpy", label="🎤 Rekam Pertanyaan Anda")
            submit_btn = gr.Button("🔁 Submit")
        with gr.Column():
            audio_output = gr.Audio(type="filepath", label="🔊 Balasan dari Asisten")
            text_output = gr.Textbox(label="💬 Teks Balasan")

    submit_btn.click(
        fn=voice_chat,
        inputs=audio_input,
        outputs=[audio_output, text_output]
    )

demo.launch()
