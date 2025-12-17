# utils/audio_utils.py
import speech_recognition as sr
from pydub import AudioSegment
import os

def audio_to_text(uploaded_file):
    recognizer = sr.Recognizer()

    audio = AudioSegment.from_file(uploaded_file)
    audio = audio.set_channels(1)

    wav_path = "temp.wav"
    audio.export(wav_path, format="wav")

    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except:
            text = ""

    os.remove(wav_path)
    return text
