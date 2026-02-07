import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
DATA_PATH = r"F:\Nehaa\Landmark_Dataset"
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 12
INPUT_SIZE = 126
# =========================================


# -------- DATASET CLASS --------
class LandmarkDataset(Dataset):
    def __init__(self, csv_file, scaler=None, fit_scaler=False):
        df = pd.read_csv(csv_file)

        self.X = df.drop("label", axis=1).values.astype(np.float32)
        self.y = (df["label"].values - 1).astype(np.int64)  # labels 0–11

        if scaler:
            if fit_scaler:
                self.X = scaler.fit_transform(self.X)
            else:
                self.X = scaler.transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------- MLP MODEL --------
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
    scaler = StandardScaler()

    train_dataset = LandmarkDataset(
        f"{DATA_PATH}\\train_landmarks.csv",
        scaler=scaler,
        fit_scaler=True
    )

    val_dataset = LandmarkDataset(
        f"{DATA_PATH}\\val_landmarks.csv",
        scaler=scaler,
        fit_scaler=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -------- MODEL --------
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # -------- TRAINING LOOP --------
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # ---- TRAIN ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ---- VALIDATION ----
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                running_loss += loss.item() * X.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss = running_loss / total
        val_acc = correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # -------- SAVE MODEL --------
    torch.save(model.state_dict(), "mlp_landmark_model.pth")
    print("\n✅ Model saved as mlp_landmark_model.pth")

    # -------- PLOTS --------
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
