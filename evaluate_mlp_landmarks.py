import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ================= CONFIG =================
DATA_PATH = r"F:\Nehaa\Landmark_Dataset"
MODEL_PATH = "mlp_landmark_model.pth"
NUM_CLASSES = 12
INPUT_SIZE = 126
# =========================================


# -------- MLP MODEL (SAME AS TRAINING) --------
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------- LOAD DATA --------
    train_df = pd.read_csv(f"{DATA_PATH}\\train_landmarks.csv")
    test_df = pd.read_csv(f"{DATA_PATH}\\test_landmarks.csv")

    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    X_test = test_df.drop("label", axis=1).values.astype(np.float32)

    y_test = (test_df["label"].values - 1).astype(np.int64)

    # -------- SCALE USING TRAIN DATA --------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_test_tensor = torch.tensor(X_test).to(device)

    # -------- LOAD MODEL --------
    model = MLP().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully")

    # -------- INFERENCE --------
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)

    y_pred = preds.cpu().numpy()

    # -------- METRICS --------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n===== TEST SET RESULTS =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # -------- CONFUSION MATRIX --------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(i) for i in range(1, 13)],
        yticklabels=[str(i) for i in range(1, 13)]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix – Landmark-Based MLP")
    plt.tight_layout()
    plt.show()

    # -------- ACCURACY GRAPH --------
    plt.figure(figsize=(4, 4))
    plt.bar(["Test Accuracy"], [accuracy])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Test Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
