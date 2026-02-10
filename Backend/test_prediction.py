
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Config (Copied from app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "mlp_landmark_model.pth")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "Landmark_Dataset", "train_landmarks.csv")
INPUT_SIZE = 126
NUM_CLASSES = 12

# Model (Copied from app.py)
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

try:
    print("Loading scaler...")
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found")
        exit(1)
        
    df = pd.read_csv(TRAIN_CSV)
    print(f"CSV Shape: {df.shape}")
    # Expected: (rows, 126 + 1)
    
    scaler = StandardScaler().fit(
        df.drop("label", axis=1).values.astype(np.float32)
    )
    print("Scaler loaded.")

    print("Loading model...")
    device = torch.device("cpu")
    model = MLP().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")

    print("Testing prediction with zeros...")
    dummy_input = np.zeros(126, dtype=np.float32)
    x = scaler.transform([dummy_input])
    x_tensor = torch.from_numpy(x).to(device)
    
    with torch.no_grad():
        output = model(x_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
    print(f"Prediction success: Class {pred.item()}, Conf {conf.item()}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
