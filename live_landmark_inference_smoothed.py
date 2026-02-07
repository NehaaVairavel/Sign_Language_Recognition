# ========== LOAD ENV FIRST ==========
from dotenv import load_dotenv
import os
load_dotenv()
# ===================================

import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque, Counter
import threading
import tempfile
import time

# Google TTS
from google.cloud import texttospeech
import sounddevice as sd
from scipy.io.wavfile import read

# PIL for Tamil text rendering
from PIL import Image, ImageDraw, ImageFont

# ================= CONFIG =================
MODEL_PATH = "mlp_landmark_model.pth"
TRAIN_CSV = r"D:\Nehaa\Project\Sign Language\Landmark_Dataset\train_landmarks.csv"
FONT_PATH = r"D:\Nehaa\Project\Sign Language\Fonts\NotoSansTamil-Regular.ttf"

INPUT_SIZE = 126
NUM_CLASSES = 12
SMOOTHING_WINDOW = 20
CONFIDENCE_THRESHOLD = 0.75
NO_HAND_RESET_FRAMES = 5
# =========================================

TAMIL_LABELS = {
    0: "அ", 1: "ஆ", 2: "இ", 3: "ஈ",
    4: "உ", 5: "ஊ", 6: "எ", 7: "ஏ",
    8: "ஐ", 9: "ஒ", 10: "ஓ", 11: "ஔ"
}

# -------- MLP MODEL (UNCHANGED) --------
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


def normalize_landmarks(lm):
    return lm - lm[0]


def sort_hands_by_x(hands):
    hands = [(h.landmark[0].x, h) for h in hands]
    hands.sort(key=lambda x: x[0])
    return [h[1] for h in hands]


# ================= AUDIO SYSTEM =================
tts_client = texttospeech.TextToSpeechClient()
audio_lock = threading.Lock()
audio_cache = {}

def play_audio(path):
    with audio_lock:
        rate, data = read(path)
        if data.ndim > 1:
            data = data[:, 0]
        sd.play(data, rate)
        sd.wait()


def speak_tamil_async(text):
    if text in audio_cache:
        threading.Thread(target=play_audio, args=(audio_cache[text],), daemon=True).start()
        return

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

    audio_cache[text] = tmp.name
    threading.Thread(target=play_audio, args=(tmp.name,), daemon=True).start()


# ================= MAIN =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load scaler
    df = pd.read_csv(TRAIN_CSV)
    scaler = StandardScaler().fit(
        df.drop("label", axis=1).values.astype(np.float32)
    )

    # Load model
    model = MLP().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully")

    # Load font
    font = ImageFont.truetype(FONT_PATH, 64)

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2)

    cap = cv2.VideoCapture(0)

    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    last_spoken = None
    no_hand_counter = 0
    current_letter = None

    print("📸 Tamil Sign → Display + Speech (Press q to quit)")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            valid_frame = False
            current_letter = None

            if res.multi_hand_landmarks and len(res.multi_hand_landmarks) == 2:
                ordered = sort_hands_by_x(res.multi_hand_landmarks)

                lm = []
                for h in ordered:
                    arr = np.array([[p.x, p.y, p.z] for p in h.landmark], np.float32)
                    lm.append(normalize_landmarks(arr))

                features = np.concatenate(lm).flatten()
                x = scaler.transform([features]).astype(np.float32)
                x = torch.from_numpy(x).to(device)

                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)

                if confidence.item() >= CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(pred.item())
                    stable = Counter(prediction_buffer).most_common(1)[0][0]
                    current_letter = TAMIL_LABELS[stable]

                    if stable != last_spoken:
                        speak_tamil_async(current_letter)
                        last_spoken = stable

                    valid_frame = True

            if not valid_frame:
                no_hand_counter += 1
                if no_hand_counter >= NO_HAND_RESET_FRAMES:
                    prediction_buffer.clear()
                    last_spoken = None
                    current_letter = None
            else:
                no_hand_counter = 0

            # -------- DISPLAY TAMIL LETTER --------
            if current_letter:
                pil_img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(pil_img)
                draw.text((40, 40), current_letter, font=font, fill=(0, 255, 0))
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.imshow("Tamil Sign Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
