import joblib
import shap
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "model", "clinical_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "..", "utils", "scaler.pkl"))

# dummy background data
X_background = np.random.normal(0, 1, (100, scaler.n_features_in_))

explainer = shap.Explainer(model.predict_proba, X_background)

joblib.dump(explainer, os.path.join(BASE_DIR, "shap_explainer.pkl"))

print("✅ SHAP explainer saved")
