# 🫀 Heart Risk Prediction System (CardioFusion-X)

An advanced **AI-powered cardiovascular risk prediction system** that combines clinical data and physiological signals to provide **accurate, explainable, and real-time heart risk assessment**.

---

## 🚀 Overview

Cardiovascular diseases are one of the leading causes of death globally. Early detection is crucial, but traditional systems rely on limited data and provide only binary outputs.

This project introduces a **multimodal heart risk prediction system** that:

- Uses **Machine Learning + Deep Learning**
- Combines **clinical + physiological (PPG) data**
- Provides **continuous risk percentage**
- Includes **Explainable AI (SHAP)**
- Offers a **visual interactive dashboard**

---

## 🧠 Key Features

- 🔬 **Multimodal Prediction**
  - Clinical Model (ML)
  - PPG Model (Deep Learning)
  - Fusion Neural Network

- 📊 **Continuous Risk Output**
  - Example: `36.65% risk`
  - Categorized into Low / Moderate / High

- 🔍 **Explainable AI (SHAP)**
  - Feature impact visualization
  - Understand *why* a prediction was made

- 📈 **Interactive UI**
  - Risk gauge (speedometer)
  - Color-coded severity
  - Real-time results

- ⚡ **Demo Mode**
  - One-click Low / Moderate / High test inputs

---


---

## 📊 Datasets Used

1. **Cardiovascular Disease Dataset (Kaggle)**  
   - ~70,000 patient records  
   - Clinical + lifestyle features  

2. **Framingham Heart Study Dataset**  
   - Long-term cardiovascular risk data  

3. **UCI Heart Disease Dataset**  
   - Standard benchmark dataset  

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/stud-addy-vans/HeartRisk-App.git
cd HeartRiskApp/backend
```

## Create virtual environment (recommended)
```
python -m venv venv
venv\Scripts\activate   # Windows
```
---

## Run Backend
```
python app.py
```

## Run Frontend
```
frontend/index.html
```
---

## 🧪 Usage

1. Enter patient clinical values (age, cholesterol, glucose, BMI, etc.) in the frontend form.  
2. The backend computes engineered biomarkers (Pulse Pressure, MRC, ABPI).  
3. The fusion model generates a continuous risk percentage.  
4. SHAP explainability highlights the most influential features.  
5. The dashboard displays risk level (Low / Moderate / High) with interactive visuals.  

---


---

## 🤝 Contributing

Contributions are welcome!  
- Fork the repository  
- Create a new branch (`feature-xyz`)  
- Commit changes and open a Pull Request  

---

## 📜 License

This project is released under the **MIT License**. You are free to use, modify, and distribute with proper attribution.  

---

## 📧 Contact

For questions or collaboration:  
- **Aditya Pratap Singh** – adityapratap2301@gmail.com  

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{HeartRiskApp2026,
  author       = {Aditya Pratap Singh and Pooja Rai},
  title        = {Heart Risk Prediction System (CardioFusion-X)},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/stud-addy-vans/HeartRisk-App}}
}

