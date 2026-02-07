import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# ================= CONFIG =================
BASE_PATH = r"F:\Nehaa\Processed_Dataset"
SPLITS = ["train", "val", "test"]
CLASSES = [str(i) for i in range(1, 13)]
OUTPUT_DIR = r"F:\Nehaa\Landmark_Dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist (index 0)
    """
    wrist = landmarks[0]
    return landmarks - wrist


def extract_two_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
        return None

    all_landmarks = []

    for hand_landmarks in results.multi_hand_landmarks:
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )
        landmarks = normalize_landmarks(landmarks)
        all_landmarks.append(landmarks)

    combined = np.concatenate(all_landmarks, axis=0)  # (42, 3)
    return combined.flatten()  # (126,)


def process_split(split):
    data = []
    labels = []

    print(f"\nProcessing {split} data...")

    for cls in CLASSES:
        class_path = os.path.join(BASE_PATH, split, cls)
        images = os.listdir(class_path)

        for img_name in tqdm(images, desc=f"{split}/{cls}", leave=False):
            img_path = os.path.join(class_path, img_name)
            features = extract_two_hand_landmarks(img_path)

            if features is None:
                continue

            data.append(features)
            labels.append(int(cls))

    df = pd.DataFrame(data)
    df["label"] = labels

    csv_path = os.path.join(OUTPUT_DIR, f"{split}_landmarks.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved {split} landmarks → {csv_path}")


def main():
    for split in SPLITS:
        process_split(split)

    print("\n✅ Landmark extraction completed for all splits.")


if __name__ == "__main__":
    main()
