// const form = document.getElementById("riskForm");
// const resultBox = document.getElementById("result");

// function calculateBMI() {
//   const h = height.value / 100;
//   const w = weight.value;

//   if (!h || !w) {
//     document.getElementById("bmiValue").innerText = "--";
//     document.getElementById("bmiStatus").innerText = "";
//     return 0;
//   }

//   const bmi = w / (h * h);

//   document.getElementById("bmiValue").innerText = bmi.toFixed(2);

//   const status = document.getElementById("bmiStatus");

//   if (bmi < 18.5) {
//     status.innerText = "Underweight";
//     status.style.color = "#00bcd4";
//   } else if (bmi < 25) {
//     status.innerText = "Normal";
//     status.style.color = "#2ecc71";
//   } else if (bmi < 30) {
//     status.innerText = "Overweight";
//     status.style.color = "#f1c40f";
//   } else {
//     status.innerText = "Obese";
//     status.style.color = "#e74c3c";
//   }

//   return Number(bmi.toFixed(2));
// }

// height.addEventListener("input", calculateBMI);
// weight.addEventListener("input", calculateBMI);

// function updateGauge(risk) {
//   risk = Math.max(0, Math.min(100, risk));

//   const needle = document.getElementById("needle");
//   const value = document.getElementById("gaugeValue");

//   let angle;

//   if (risk <= 30) {
//     // 0–30% → -90° to -36°
//     angle = -90 + (risk / 30) * 54;
//   } else if (risk <= 60) {
//     // 30–60% → -36° to +18°
//     angle = -36 + ((risk - 30) / 30) * 54;
//   } else {
//     // 60–100% → +18° to +90°
//     angle = 18 + ((risk - 60) / 40) * 72;
//   }

//   needle.setAttribute("transform", `rotate(${angle} 160 160)`);

//   value.innerText = risk.toFixed(2) + "%";
// }

// form.addEventListener("submit", async (e) => {
//   e.preventDefault();

//   const data = {
//     age: +age.value,
//     gender: +gender.value,
//     bp: +bp.value,
//     bmi: parseFloat(calculateBMI()),
//     cholesterol: +cholesterol.value,
//     glucose: +glucose.value,
//     smoking: +smoking.value,
//     alcohol: +alcohol.value,
//     activity: +activity.value,

//     heart_rate: +heart_rate.value,
//     spo2: +spo2.value,
//     hrv: +hrv.value,
//     pulse_amplitude: +pulse_amplitude.value,
//   };

//   const res = await fetch("http://127.0.0.1:5000/predict", {
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify(data),
//   });

//   const result = await res.json();

//   document.getElementById("clinicalRisk").innerText = result.clinical_risk;
//   document.getElementById("ppgRisk").innerText = result.ppg_risk;
//   document.getElementById("finalRisk").innerText = result.final_risk;

//   const riskLevel = document.getElementById("riskLevel");
//   const risk = result.final_risk;

//   if (risk < 30) {
//     riskLevel.innerText = "LOW RISK";
//     riskLevel.style.background = "#2ecc71";
//   } else if (risk < 60) {
//     riskLevel.innerText = "MODERATE RISK";
//     riskLevel.style.background = "#f1c40f";
//   } else {
//     riskLevel.innerText = "HIGH RISK";
//     riskLevel.style.background = "#e74c3c";
//   }

//   resultBox.classList.remove("hidden");
//   updateGauge(result.final_risk);

//   const explainList = document.getElementById("explainList");
//   explainList.innerHTML = "";

//   if (result.explanations && result.explanations.length > 0) {
//     result.explanations.forEach((item) => {
//       const li = document.createElement("li");

//       const arrow = item.impact > 0 ? "↑" : "↓";
//       const color = item.impact > 0 ? "#e74c3c" : "#2ecc71";

//       li.innerHTML = `
//       <span style="color:${color}; font-weight:bold">${arrow}</span>
//       ${item.text}
//     `;

//       explainList.appendChild(li);
//     });
//   } else {
//     explainList.innerHTML = "<li>No explanation available</li>";
//   }
// });

const form = document.getElementById("riskForm");
const resultBox = document.getElementById("result");
const FEATURE_BASELINE = {
  Age: 50,
  BMI: 24,
  "Blood Pressure": 130,
  Cholesterol: 220,
  Glucose: 2,
  Smoking: 0.3,
  Alcohol: 0.2,
  "Physical Activity": 0.4,
  Gender: 0.5,
};

const FEATURE_LABELS = {
  Age: "Age",
  BMI: "Body mass index",
  "Blood Pressure": "Blood pressure",
  Cholesterol: "Cholesterol",
  Glucose: "Blood glucose",
  Smoking: "Smoking habit",
  Alcohol: "Alcohol intake",
  "Physical Activity": "Physical activity",
  Gender: "Gender",
};

// -----------------------------
// BMI calculation
// -----------------------------
function calculateBMI() {
  const h = height.value / 100;
  const w = weight.value;

  if (!h || !w) {
    bmiValue.innerText = "--";
    bmiStatus.innerText = "";
    return 0;
  }

  const bmi = w / (h * h);
  bmiValue.innerText = bmi.toFixed(2);

  if (bmi < 18.5) {
    bmiStatus.innerText = "Underweight";
    bmiStatus.style.color = "#00bcd4";
  } else if (bmi < 25) {
    bmiStatus.innerText = "Normal";
    bmiStatus.style.color = "#2ecc71";
  } else if (bmi < 30) {
    bmiStatus.innerText = "Overweight";
    bmiStatus.style.color = "#f1c40f";
  } else {
    bmiStatus.innerText = "Obese";
    bmiStatus.style.color = "#e74c3c";
  }

  return Number(bmi.toFixed(2));
}

// ----------------------------------
// Convert SHAP value to human text
// ----------------------------------
function explainFeature(feature, impact, inputData) {
  const baseline = FEATURE_BASELINE[feature];
  const value = inputData[feature];

  if (baseline === undefined || value === undefined) {
    return `${feature} influenced the prediction.`;
  }

  const diff = value - baseline;
  const absImpact = Math.abs(impact);

  let strength = "minor";

  if (absImpact > 0.25) strength = "strong";
  else if (absImpact > 0.12) strength = "moderate";

  if (diff > 0) {
    return `${FEATURE_LABELS[feature]} is above average → ${strength} influence on risk.`;
  } else {
    return `${FEATURE_LABELS[feature]} is below average → ${strength} influence on risk.`;
  }
}

function generateRecommendations(input, finalRisk) {
  const advice = [];

  // -----------------------
  // Risk-based advice
  // -----------------------
  if (finalRisk < 30) {
    advice.push(
      "Maintain your healthy lifestyle and continue routine health checkups.",
    );
  } else if (finalRisk < 60) {
    advice.push(
      "Moderate heart risk detected. Lifestyle improvements are strongly advised.",
    );
  } else {
    advice.push(
      "High cardiovascular risk detected. Medical consultation is recommended.",
    );
  }

  // -----------------------
  // Clinical rules
  // -----------------------
  if (input.bp > 140) {
    advice.push(
      "Blood pressure is elevated. Reduce salt intake and monitor BP regularly.",
    );
  }

  if (input.bmi >= 25) {
    advice.push(
      "Body weight is above recommended range. Weight reduction may lower heart risk.",
    );
  }

  if (input.cholesterol > 240) {
    advice.push(
      "High cholesterol detected. A low-fat diet and lipid profile monitoring are advised.",
    );
  }

  if (input.glucose >= 3) {
    advice.push(
      "Elevated blood glucose observed. Regular sugar monitoring is recommended.",
    );
  }

  if (input.smoking === 1) {
    advice.push(
      "Smoking significantly increases cardiovascular risk. Smoking cessation is strongly advised.",
    );
  }

  if (input.alcohol === 1) {
    advice.push("Limiting alcohol consumption may help reduce heart risk.");
  }

  if (input.activity === 0) {
    advice.push(
      "Low physical activity detected. At least 30 minutes of daily exercise is recommended.",
    );
  }

  // -----------------------
  // PPG-based
  // -----------------------
  if (input.heart_rate > 100) {
    advice.push(
      "Elevated resting heart rate observed. Consider cardiovascular fitness evaluation.",
    );
  }

  if (input.spo2 < 95) {
    advice.push(
      "Low oxygen saturation detected. Respiratory evaluation may be beneficial.",
    );
  }

  if (input.hrv < 30) {
    advice.push(
      "Low heart rate variability may indicate stress or fatigue. Adequate rest is advised.",
    );
  }

  return advice;
}

height.addEventListener("input", calculateBMI);
weight.addEventListener("input", calculateBMI);

// -----------------------------
// Gauge update
// -----------------------------
function updateGauge(risk) {
  risk = Math.max(0, Math.min(100, risk));

  let angle;
  if (risk <= 30) {
    angle = -90 + (risk / 30) * 54;
  } else if (risk <= 60) {
    angle = -36 + ((risk - 30) / 30) * 54;
  } else {
    angle = 18 + ((risk - 60) / 40) * 72;
  }

  needle.setAttribute("transform", `rotate(${angle} 160 160)`);
  gaugeValue.innerText = risk.toFixed(2) + "%";
}

// -----------------------------
// Form submit
// -----------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const data = {
    age: +age.value,
    gender: +gender.value,
    bp: +bp.value,
    bmi: calculateBMI(),
    cholesterol: +cholesterol.value,
    glucose: +glucose.value,
    smoking: +smoking.value,
    alcohol: +alcohol.value,
    activity: +activity.value,

    heart_rate: +heart_rate.value,
    spo2: +spo2.value,
    hrv: +hrv.value,
    pulse_amplitude: +pulse_amplitude.value,
  };

  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  const result = await res.json();

  // -----------------------------
  // Numeric results
  // -----------------------------
  clinicalRisk.innerText = result.clinical_risk;
  ppgRisk.innerText = result.ppg_risk;
  finalRisk.innerText = result.final_risk;

  // -----------------------------
  // Risk level label
  // -----------------------------
  const risk = result.final_risk;

  if (risk < 30) {
    riskLevel.innerText = "LOW RISK";
    riskLevel.style.background = "#2ecc71";
  } else if (risk < 60) {
    riskLevel.innerText = "MODERATE RISK";
    riskLevel.style.background = "#f1c40f";
  } else {
    riskLevel.innerText = "HIGH RISK";
    riskLevel.style.background = "#e74c3c";
  }

  // -----------------------------
  // Gauge
  // -----------------------------
  updateGauge(risk);
  resultBox.classList.remove("hidden");

  // -----------------------------
  // SHAP explanations
  // -----------------------------
  const explainList = document.getElementById("explainList");
  explainList.innerHTML = "";

  if (result.explanations && result.explanations.length > 0) {
    result.explanations.forEach((item) => {
      const li = document.createElement("li");

      const arrow = item.impact > 0 ? "↑" : "↓";
      const color = item.impact > 0 ? "#e74c3c" : "#2ecc71";

      const explanationText = explainFeature(item.feature, item.impact, {
        Age: +age.value,
        BMI: calculateBMI(),
        "Blood Pressure": +bp.value,
        Cholesterol: +cholesterol.value,
        Glucose: +glucose.value,
        Smoking: +smoking.value,
        Alcohol: +alcohol.value,
        "Physical Activity": +activity.value,
        Gender: +gender.value,
      });

      li.innerHTML = `
  <span style="color:${color}; font-weight:bold">${arrow}</span>
  ${explanationText}
`;

      explainList.appendChild(li);
    });
  } else {
    explainList.innerHTML = "<li>No explanation available</li>";
  }

  const recommendList = document.getElementById("recommendList");
  recommendList.innerHTML = "";

  const recommendations = generateRecommendations(
    {
      age: +age.value,
      bmi: calculateBMI(),
      bp: +bp.value,
      cholesterol: +cholesterol.value,
      glucose: +glucose.value,
      smoking: +smoking.value,
      alcohol: +alcohol.value,
      activity: +activity.value,
      heart_rate: +heart_rate.value,
      spo2: +spo2.value,
      hrv: +hrv.value,
    },
    result.final_risk,
  );

  recommendations.forEach((text) => {
    const li = document.createElement("li");
    li.innerText = text;
    recommendList.appendChild(li);
  });
});

function fillDemo(type) {
  if (type === "low") {
    age.value = 22;
    gender.value = 1;
    height.value = 175;
    weight.value = 65;
    bp.value = 110;
    cholesterol.value = 170;
    glucose.value = 1;
    smoking.value = 0;
    alcohol.value = 0;
    activity.value = 1;

    heart_rate.value = 65;
    spo2.value = 98;
    hrv.value = 70;
    pulse_amplitude.value = 1.3;
  }

  if (type === "mid") {
    age.value = 38;
    gender.value = 1;
    height.value = 168;
    weight.value = 78;
    bp.value = 135;
    cholesterol.value = 200;
    glucose.value = 2;
    smoking.value = 0;
    alcohol.value = 0;
    activity.value = 1;

    heart_rate.value = 82;
    spo2.value = 93;
    hrv.value = 35;
    pulse_amplitude.value = 0.9;
  }

  if (type === "high") {
    age.value = 60;
    gender.value = 1;
    height.value = 168;
    weight.value = 95;
    bp.value = 175;
    cholesterol.value = 290;
    glucose.value = 3;
    smoking.value = 1;
    alcohol.value = 1;
    activity.value = 0;

    heart_rate.value = 90;
    spo2.value = 92;
    hrv.value = 20;
    pulse_amplitude.value = 0.3;
  }

  calculateBMI();
}

document.getElementById("lowBtn").onclick = () => fillDemo("low");
document.getElementById("midBtn").onclick = () => fillDemo("mid");
document.getElementById("highBtn").onclick = () => fillDemo("high");
