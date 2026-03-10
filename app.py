import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(page_title="Heart Stroke AI Predictor", page_icon="❤️", layout="wide")

# -------------------------------
# Custom CSS Styling
# -------------------------------

st.markdown(
    """
<style>

.main-title{
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:white;
}

.sub-text{
    text-align:center;
    color:white;
    font-size:18px;
}

.header{
    background: linear-gradient(90deg,#ff4b2b,#ff416c);
    padding:30px;
    border-radius:12px;
}

.stButton>button{
    background-color:#ff4b2b;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:3em;
    width:100%;
}

.result-success{
    padding:20px;
    border-radius:10px;
    background-color:#e8f5e9;
    color:#2e7d32;
    font-size:22px;
    text-align:center;
}

.result-danger{
    padding:20px;
    border-radius:10px;
    background-color:#ffebee;
    color:#c62828;
    font-size:22px;
    text-align:center;
}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("🫀 Heart Health AI")

st.sidebar.markdown("""
### About This App

This AI system predicts **Heart Disease Risk** using Machine Learning.

### Model Used
K-Nearest Neighbors (KNN)

### Features
- Age
- Cholesterol
- Blood Pressure
- Heart Rate
- ECG Results
- Exercise Angina
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### Developer")
st.sidebar.write("Aarush Rawat")

# -------------------------------
# Header Section
# -------------------------------

st.markdown(
    """
    <div class="header">
        <div class="main-title">❤️ Heart Stroke Prediction AI</div>
        <div class="sub-text">
        Machine Learning Powered Heart Risk Assessment Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")

# -------------------------------
# Dashboard Metrics
# -------------------------------

m1, m2 = st.columns(2)

m1.metric("Features", "11")
m2.metric("Prediction Type", "Binary Classification")

st.write("")
st.write("")

# -------------------------------
# Load Model Files
# -------------------------------

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# -------------------------------
# Input Form
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write("")
st.write("")

# -------------------------------
# Prediction
# -------------------------------

if st.button("🔍 Predict Heart Disease Risk"):
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    risk_percentage = round(probability * 100, 2)

    st.write("")
    st.subheader("🧠 Prediction Result")

    # -------------------------------
    # Risk Gauge Chart
    # -------------------------------

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            title={"text": "Heart Disease Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "salmon"},
                ],
            },
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Result Message
    # -------------------------------

    if prediction == 1:
        st.markdown(
            f"<div class='result-danger'>⚠️ High Risk of Heart Disease ({risk_percentage}%)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='result-success'>✅ Low Risk of Heart Disease ({risk_percentage}%)</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------
    # Risk Chart
    # -------------------------------

    st.write("")
    st.subheader("📊 Risk Visualization")

    fig2, ax = plt.subplots()

    ax.bar(["Risk Probability"], [risk_percentage])
    ax.set_ylabel("Risk %")
    ax.set_ylim(0, 100)

    st.pyplot(fig2)

    # -------------------------------
    # Health Recommendations
    # -------------------------------
    st.write("")
    st.subheader("🩺 Health Recommendations")

    if prediction == 1:
        st.warning("""
- Consult a cardiologist
- Reduce cholesterol intake
- Exercise regularly
- Monitor blood pressure
- Avoid smoking
""")
    else:
        st.info("""
- Maintain healthy diet
- Regular physical activity
- Routine health checkups
- Manage stress levels
""")
