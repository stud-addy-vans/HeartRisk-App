import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Generate synthetic PPG data
# -----------------------------
np.random.seed(42)

samples = 70000

heart_rate = np.random.normal(75, 15, samples)
spo2 = np.random.normal(97, 2, samples)
hrv = np.random.normal(50, 20, samples)
pulse_amp = np.random.normal(1.0, 0.3, samples)

X = np.column_stack([heart_rate, spo2, hrv, pulse_amp])

# -----------------------------
# Create synthetic risk label
# -----------------------------
risk = (
    (heart_rate > 95).astype(int) +
    (spo2 < 92).astype(int) +
    (hrv < 30).astype(int) +
    (pulse_amp < 0.6).astype(int)
)

y = (risk >= 2).astype(int)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(MODEL_DIR, "ppg_scaler.pkl"))

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Build neural network
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train
# -----------------------------
print("⏳ Training PPG neural network...")

model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# Save model
# -----------------------------
model.save(os.path.join(MODEL_DIR, "ppg_model.h5"))

print("✅ PPG model saved successfully")
