import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UTILS_DIR = os.path.join(BASE_DIR, "utils")

os.makedirs(UTILS_DIR, exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
cardio = pd.read_csv(os.path.join(DATA_DIR, "cardio_train.csv"), sep=";")
heart = pd.read_csv(os.path.join(DATA_DIR, "heart_disease.csv"))
uci = pd.read_csv(os.path.join(DATA_DIR, "uci_heart.csv"))

# -----------------------------
# Fix UCI missing values
# -----------------------------
uci = uci.replace("?", np.nan)

# -----------------------------
# CARDIO DATASET
# -----------------------------
cardio["age"] = cardio["age"] / 365
cardio["gender"] = cardio["gender"].map({1: 0, 2: 1})
cardio["bmi"] = cardio["weight"] / ((cardio["height"] / 100) ** 2)

cardio_df = cardio[[
    "age",
    "gender",
    "bmi",
    "ap_hi",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio"
]].rename(columns={
    "ap_hi": "bp",
    "gluc": "glucose",
    "cardio": "target"
})

# -----------------------------
# HEART DISEASE DATASET
# -----------------------------
heart_df = heart.rename(columns={
    "Age": "age",
    "Sex": "gender",
    "RestingBP": "bp",
    "Cholesterol": "cholesterol",
    "FastingBS": "glucose",
    "HeartDisease": "target"
})

heart_df["bmi"] = np.nan
heart_df["smoking"] = np.nan
heart_df["alcohol"] = np.nan
heart_df["activity"] = 1

heart_df = heart_df[[
    "age", "gender", "bmi", "bp",
    "cholesterol", "glucose",
    "smoking", "alcohol",
    "activity", "target"
]]

# -----------------------------
# UCI DATASET
# -----------------------------
uci_df = uci.rename(columns={
    "age": "age",
    "sex": "gender",
    "trestbps": "bp",
    "chol": "cholesterol",
    "fbs": "glucose",
    "condition": "target"
})

uci_df["bmi"] = np.nan
uci_df["smoking"] = np.nan
uci_df["alcohol"] = np.nan
uci_df["activity"] = 1

uci_df = uci_df[[
    "age", "gender", "bmi", "bp",
    "cholesterol", "glucose",
    "smoking", "alcohol",
    "activity", "target"
]]

# -----------------------------
# Merge datasets
# -----------------------------
final_df = pd.concat([cardio_df, heart_df, uci_df], ignore_index=True)

# -----------------------------
# Convert all columns to numeric
# -----------------------------
final_df = final_df.apply(pd.to_numeric, errors="coerce")

# -----------------------------
# Fill missing values
# -----------------------------
final_df = final_df.fillna(final_df.median())

# -----------------------------
# Split features & target
# -----------------------------
X = final_df.drop("target", axis=1)
y = final_df["target"]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(UTILS_DIR, "scaler.pkl"))

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Save arrays
# -----------------------------
np.save(os.path.join(BASE_DIR, "training", "X_train.npy"), X_train)
np.save(os.path.join(BASE_DIR, "training", "X_test.npy"), X_test)
np.save(os.path.join(BASE_DIR, "training", "y_train.npy"), y_train)
np.save(os.path.join(BASE_DIR, "training", "y_test.npy"), y_test)

print("✅ Preprocessing completed successfully")
print("Final dataset shape:", final_df.shape)
