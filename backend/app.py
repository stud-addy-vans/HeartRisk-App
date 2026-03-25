# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import joblib
# import tensorflow as tf
# import os

# # -------------------------------------------------
# # App setup
# # -------------------------------------------------
# app = Flask(__name__)
# CORS(app)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # -------------------------------------------------
# # Load SHAP explainer
# # -------------------------------------------------
# shap_explainer = joblib.load(
#     os.path.join(BASE_DIR, "explain", "shap_explainer.pkl")
# )

# # -------------------------------------------------
# # Load models and scalers
# # -------------------------------------------------
# clinical_model = joblib.load(
#     os.path.join(BASE_DIR, "model", "clinical_model.pkl")
# )

# scaler = joblib.load(
#     os.path.join(BASE_DIR, "utils", "scaler.pkl")
# )

# ppg_model = tf.keras.models.load_model(
#     os.path.join(BASE_DIR, "model", "ppg_model.h5")
# )

# ppg_scaler = joblib.load(
#     os.path.join(BASE_DIR, "utils", "ppg_scaler.pkl")
# )

# fusion_model = tf.keras.models.load_model(
#     os.path.join(BASE_DIR, "model", "fusion_model.h5")
# )

# # -------------------------------------------------
# # Feature names
# # -------------------------------------------------
# FEATURE_NAMES = [
#     "Age",
#     "Gender",
#     "BMI",
#     "Blood Pressure",
#     "Cholesterol",
#     "Glucose",
#     "Smoking",
#     "Alcohol",
#     "Physical Activity"
# ]

# # -------------------------------------------------
# # Strength mapping
# # -------------------------------------------------
# def explain_strength(value):
#     value = abs(value)

#     if value < 0.05:
#         return "very small"
#     elif value < 0.12:
#         return "slight"
#     elif value < 0.25:
#         return "moderate"
#     else:
#         return "strong"


# MEDICAL_NAMES = {
#     "Age": "Age",
#     "BMI": "Body mass index",
#     "Blood Pressure": "Blood pressure",
#     "Cholesterol": "Cholesterol level",
#     "Glucose": "Blood glucose",
#     "Smoking": "Smoking habit",
#     "Alcohol": "Alcohol intake",
#     "Physical Activity": "Physical activity"
# }

# # -------------------------------------------------
# # Prediction endpoint
# # -------------------------------------------------
# @app.route("/predict", methods=["POST"])
# def predict():

#     data = request.json

#     # -------------------------------------------------
#     # Clinical features
#     # -------------------------------------------------
#     clinical_features = np.array([[
#         data["age"],
#         data["gender"],
#         data["bmi"],
#         data["bp"],
#         data["cholesterol"],
#         data["glucose"],
#         data["smoking"],
#         data["alcohol"],
#         data["activity"]
#     ]])

#     # pad to scaler size
#     clinical_padded = np.pad(
#         clinical_features,
#         ((0, 0), (0, scaler.n_features_in_ - clinical_features.shape[1])),
#         mode="constant"
#     )

#     clinical_scaled = scaler.transform(clinical_padded)

#     clinical_risk = clinical_model.predict_proba(
#         clinical_scaled
#     )[0][1]

#     # -------------------------------------------------
#     # SHAP EXPLANATION (stable)
#     # -------------------------------------------------
#     shap_values = shap_explainer.shap_values(clinical_scaled)

#     shap_scores = shap_values[1][0]
#     shap_scores = np.round(shap_scores, 4)

#     top_features = sorted(
#         zip(FEATURE_NAMES, shap_scores),
#         key=lambda x: abs(x[1]),
#         reverse=True
#     )[:5]

#     final_explanations = []

#     for feature, value in top_features:

#         direction = "increased" if value > 0 else "reduced"
#         strength = explain_strength(value)

#         label = MEDICAL_NAMES.get(feature, feature)

#         sentence = f"{label} {direction} heart risk ({strength} effect)."

#         final_explanations.append({
#             "feature": feature,
#             "impact": float(value),
#             "text": sentence
#         })

#     # -------------------------------------------------
#     # PPG features
#     # -------------------------------------------------
#     ppg_features = np.array([[
#         data["heart_rate"],
#         data["spo2"],
#         data["hrv"],
#         data["pulse_amplitude"]
#     ]])

#     ppg_scaled = ppg_scaler.transform(ppg_features)

#     ppg_risk = float(
#         ppg_model.predict(ppg_scaled)[0][0]
#     )

#     # -------------------------------------------------
#     # Fusion model
#     # -------------------------------------------------
#     fusion_input = np.array([[clinical_risk, ppg_risk]])

#     final_risk = float(
#         fusion_model.predict(fusion_input)[0][0]
#     )

#     # -------------------------------------------------
#     # Response
#     # -------------------------------------------------
#     return jsonify({
#         "clinical_risk": round(clinical_risk * 100, 2),
#         "ppg_risk": round(ppg_risk * 100, 2),
#         "final_risk": round(final_risk * 100, 2),
#         "explanations": final_explanations
#     })


# # -------------------------------------------------
# # Run server
# # -------------------------------------------------
# if __name__ == "__main__":
#     app.run(debug=True)




from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
import os

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# Load models
# -------------------------------------------------
clinical_model = joblib.load(
    os.path.join(BASE_DIR, "model", "clinical_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "utils", "scaler.pkl")
)

ppg_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model", "ppg_model.h5")
)

ppg_scaler = joblib.load(
    os.path.join(BASE_DIR, "utils", "ppg_scaler.pkl")
)

fusion_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model", "fusion_model.h5")
)

shap_explainer = joblib.load(
    os.path.join(BASE_DIR, "explain", "shap_explainer.pkl")
)

# -------------------------------------------------
# Feature names
# -------------------------------------------------
FEATURE_NAMES = [
    "Age",
    "Gender",
    "BMI",
    "Blood Pressure",
    "Cholesterol",
    "Glucose",
    "Smoking",
    "Alcohol",
    "Physical Activity"
]

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # ---------------------------
    # Clinical features
    # ---------------------------
    clinical_features = np.array([[

        data["age"],
        data["gender"],
        data["bmi"],
        data["bp"],
        data["cholesterol"],
        data["glucose"],
        data["smoking"],
        data["alcohol"],
        data["activity"]

    ]])

    clinical_padded = np.pad(
        clinical_features,
        ((0, 0), (0, scaler.n_features_in_ - clinical_features.shape[1])),
        mode="constant"
    )

    clinical_scaled = scaler.transform(clinical_padded)

    clinical_risk = float(
        clinical_model.predict_proba(clinical_scaled)[0][1]
    )

    # ---------------------------
    # SHAP explanation
    # ---------------------------
    shap_values = shap_explainer.shap_values(clinical_scaled)
    shap_scores = shap_values[0].flatten()

    explanations = sorted(
        zip(FEATURE_NAMES, shap_scores),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )[:5]

    explanation_output = []

    for feature, value in explanations:
        explanation_output.append({
            "feature": feature,
            "impact": float(value)
        })

    # ---------------------------
    # PPG features
    # ---------------------------
    ppg_features = np.array([[

        data["heart_rate"],
        data["spo2"],
        data["hrv"],
        data["pulse_amplitude"]

    ]])

    ppg_scaled = ppg_scaler.transform(ppg_features)

    ppg_risk = float(
        ppg_model.predict(ppg_scaled)[0][0]
    )

    # ---------------------------
    # Fusion
    # ---------------------------
    fusion_input = np.array([[clinical_risk, ppg_risk]])

    final_risk = float(
        fusion_model.predict(fusion_input)[0][0]
    )

    # ---------------------------
    # Response
    # ---------------------------
    return jsonify({
        "clinical_risk": round(clinical_risk * 100, 2),
        "ppg_risk": round(ppg_risk * 100, 2),
        "final_risk": round(final_risk * 100, 2),
        "explanations": explanation_output
    })


if __name__ == "__main__":
    app.run()
