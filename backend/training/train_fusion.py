import numpy as np
import tensorflow as tf
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAIN_DIR = os.path.join(BASE_DIR, "training")

# -----------------------------
# Load models
# -----------------------------
clinical_model = joblib.load(os.path.join(MODEL_DIR, "clinical_model.pkl"))
ppg_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "ppg_model.h5"))

# -----------------------------
# Load training data
# -----------------------------
X_train = np.load(os.path.join(TRAIN_DIR, "X_train.npy"))
y_train = np.load(os.path.join(TRAIN_DIR, "y_train.npy"))

# -----------------------------
# Generate fusion inputs
# -----------------------------
clinical_probs = clinical_model.predict_proba(X_train)[:, 1]

# Simulate PPG probabilities
ppg_probs = np.clip(
    clinical_probs + np.random.normal(0, 0.08, len(clinical_probs)),
    0, 1
)

fusion_X = np.column_stack([clinical_probs, ppg_probs])
fusion_y = y_train

# -----------------------------
# Fusion neural network
# -----------------------------
fusion_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

fusion_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("⏳ Training fusion model...")

fusion_model.fit(
    fusion_X,
    fusion_y,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# Save fusion model
# -----------------------------
fusion_model.save(os.path.join(MODEL_DIR, "fusion_model.h5"))

print("✅ Fusion model saved successfully")
