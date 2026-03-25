import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "training")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load preprocessed data
# -----------------------------
X_train = np.load(os.path.join(TRAIN_DIR, "X_train.npy"))
X_test = np.load(os.path.join(TRAIN_DIR, "X_test.npy"))
y_train = np.load(os.path.join(TRAIN_DIR, "y_train.npy"))
y_test = np.load(os.path.join(TRAIN_DIR, "y_test.npy"))

# -----------------------------
# Build Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Train model
# -----------------------------
print("⏳ Training clinical model...")
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n✅ Clinical Model Performance")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC :", round(auc * 100, 2), "%\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "clinical_model.pkl"))

print("💾 clinical_model.pkl saved successfully")
