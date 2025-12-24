import whisper
import os

# Load once (important for performance)
model = whisper.load_model("base")

def transcribe_audio(file_path):
    """
    Takes an audio file path and returns transcribed text
    """
    result = model.transcribe(file_path)
    return result["text"]