import pandas as pd
import os

DATASET_PATH = r"F:\Nehaa\Landmark_Dataset"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    path = os.path.join(DATASET_PATH, f"{split}_landmarks.csv")
    df = pd.read_csv(path)

    print(f"\n--- {split.upper()} DATASET ---")
    print("Total samples:", len(df))
    print("Missing values:", df.isnull().sum().sum())

    feature_count = df.shape[1] - 1
    print("Feature count per sample:", feature_count)

    if feature_count != 126:
        print("❌ ERROR: Feature size mismatch!")
    else:
        print("✅ Feature size correct (126)")

    print("Class distribution:")
    print(df["label"].value_counts().sort_index())
