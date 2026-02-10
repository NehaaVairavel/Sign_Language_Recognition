import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from google.cloud import texttospeech

# ================= LOAD ENV =================
# ================= LOAD ENV =================
# Load from parent directory (root)
from dotenv import load_dotenv
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ================= CONFIG =================
# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "mlp_landmark_model.pth")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "Landmark_Dataset", "train_landmarks.csv")
INPUT_SIZE = 126
NUM_CLASSES = 12
CONFIDENCE_THRESHOLD = 0.75

TAMIL_LABELS = {
    0: "அ", 1: "ஆ", 2: "இ", 3: "ஈ",
    4: "உ", 5: "ஊ", 6: "எ", 7: "ஏ",
    8: "ஐ", 9: "ஒ", 10: "ஓ", 11: "ஔ"
}

# ================= FLASK APP =================
app = Flask(__name__)
CORS(app)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL =================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ================= SCALER =================
df = pd.read_csv(TRAIN_CSV)
scaler = StandardScaler().fit(
    df.drop("label", axis=1).values.astype(np.float32)
)

# ================= TTS =================
tts_client = texttospeech.TextToSpeechClient()

# ======================================================
# API: Health Check
# ======================================================
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Tamil Sign Language Backend Running"})


# ======================================================
# API: Predict Sign (Core API)
# ======================================================
@app.route("/predict", methods=["POST"])
def predict_sign():
    try:
        data = request.json
        if "landmarks" not in data:
            return jsonify({"error": "Landmarks missing"}), 400

        # Log input shape for debugging
        landmarks_list = data["landmarks"]
        # with open("debug_input.log", "w") as f:
        #     f.write(f"Input legnth: {len(landmarks_list)}\n")
        #     f.write(str(landmarks_list[:10]))

        landmarks = np.array(landmarks_list, dtype=np.float32)

        if landmarks.shape[0] != INPUT_SIZE:
            with open("backend_error.log", "w") as f:
                f.write(f"Invalid landmark size: {landmarks.shape[0]}, expected {INPUT_SIZE}")
            return jsonify({"error": f"Invalid landmark size: {landmarks.shape[0]}"}), 400

        x = scaler.transform([landmarks])
        x = torch.from_numpy(x).float().to(device)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        confidence = confidence.item()
        pred_class = pred.item()

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "valid": False,
                "confidence": confidence
            })

        return jsonify({
            "valid": True,
            "class_id": pred_class + 1,
            "tamil_letter": TAMIL_LABELS[pred_class],
            "confidence": confidence
        })
    except Exception as e:
        import traceback
        with open("backend_error.log", "w") as f:
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ======================================================
# API: Verify Sign (Learning / Practice / Quiz)
# ======================================================
@app.route("/verify", methods=["POST"])
def verify_sign():
    data = request.json

    landmarks = np.array(data["landmarks"], dtype=np.float32)
    expected_class = int(data["expected_class"]) - 1

    x = scaler.transform([landmarks])
    x = torch.from_numpy(x).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    is_correct = (
        pred.item() == expected_class and
        confidence.item() >= CONFIDENCE_THRESHOLD
    )

    return jsonify({
        "correct": is_correct,
        "predicted_class": pred.item() + 1,
        "tamil_letter": TAMIL_LABELS[pred.item()],
        "confidence": confidence.item()
    })


# ======================================================
# API: Tamil Speech (Google TTS)
# ======================================================
@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Text missing"}), 400

    response = tts_client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code="ta-IN",
            name="ta-IN-Wavenet-A"
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=0.6
        )
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(response.audio_content)
    tmp.close()

    return send_file(
        tmp.name,
        mimetype="audio/wav",
        as_attachment=False
    )


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
