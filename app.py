# app.py
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
from flask import Flask, render_template, request, jsonify
from model.text_model import StressModel, DepressionModel, EmotionModel
from utils.audio_utils import audio_to_text
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

stress_model = StressModel()
depression_model = DepressionModel()
emotion_model = EmotionModel()

def analyze_text(text):
    stress = stress_model.predict([text])[0]
    depression = depression_model.predict([text])[0]
    emotions = emotion_model.predict([text]).iloc[0].to_dict()
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "stress": round(stress, 4),
        "depression": round(depression, 4),
        "dominant_emotion": dominant_emotion,
        "emotions": emotions
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze-text", methods=["POST"])
def analyze_text_api():
    text = request.json.get("text", "")
    return jsonify(analyze_text(text))

@app.route("/analyze-audio", methods=["POST"])
def analyze_audio_api():
    file = request.files["audio"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    text = audio_to_text(path)
    result = analyze_text(text)

    result["recognized_text"] = text
    return jsonify(result)

if __name__ == "__main__":
    # Use PORT provided by host (Render/Heroku). Default to 5000 locally.
    port = int(os.environ.get("PORT", 5000))
    # Never run debug=True in production
    app.run(host="0.0.0.0", port=port, debug=False)
