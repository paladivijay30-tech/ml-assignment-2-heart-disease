import streamlit as st
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json

# --------------------------------------------------------
# LOAD METRICS JSON
# --------------------------------------------------------
with open("model_metrics.json", "r") as f:
    model_metrics = json.load(f)

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction App",
    layout="wide",
    page_icon="❤️"
)

# --------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------
def load_css():
    st.markdown("""
    <style>
        .main {
            padding: 0rem 2rem;
        }
        .big-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .sub-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------
menu = st.sidebar.radio("Navigation", ["🏠 Home", "🔮 Predict", "📊 Model Comparison", "ℹ️ About"])

# --------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------
def load_pkl(path):
    return pickle.load(open(path, "rb"))

models = {
    "Logistic Regression": load_pkl("Logistic_Regression.pkl"),
    "Decision Tree": load_pkl("Decision_Tree.pkl"),
    "Random Forest": load_pkl("Random_Forest.pkl"),
    "XGBoost": load_pkl("XGBoost.pkl"),
    "KNN": load_pkl("KNN.pkl"),
    "Naive Bayes": load_pkl("Naive_Bayes.pkl"),
}

scaler = load_pkl("scaler.pkl")

# --------------------------------------------------------
# USER INPUT FORM
# --------------------------------------------------------
def get_user_input():
    st.markdown("<div class='sub-title'>Enter Patient Details</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 40)
        resting_bp = st.number_input("Resting BP", 50, 250, 120)
        cholesterol = st.number_input("Cholesterol", 50, 600, 200)
        max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

    with col2:
        sex = st.selectbox("Sex", ["M", "F"])
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST"])
        exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
        st_slope = st.selectbox("ST Slope", ["Flat", "Up"])

    return {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0,
    }

# --------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------
if menu == "🏠 Home":
    st.markdown("<div class='big-title'>❤️ Heart Disease Prediction App</div>", unsafe_allow_html=True)
    st.write("This application predicts heart disease using multiple ML models.")

# --------------------------------------------------------
# PREDICT PAGE
# --------------------------------------------------------
elif menu == "🔮 Predict":
    st.markdown("<div class='big-title'>❤️ Heart Disease Prediction</div>", unsafe_allow_html=True)

    input_data = get_user_input()
    df_input = pd.DataFrame([input_data])

    scaled = scaler.transform(df_input)
    best_model = models["XGBoost"]
    pred = best_model.predict(scaled)[0]

    if st.button("Predict"):
        if pred == 1:
            st.error("⚠️ High Risk of Heart Disease", icon="🚨")
        else:
            st.success("✅ Low Risk of Heart Disease", icon="💚")

# --------------------------------------------------------
# MODEL COMPARISON PAGE
# --------------------------------------------------------
elif menu == "📊 Model Comparison":

    st.markdown("<div class='big-title'>📊 Model Comparison Dashboard</div>", unsafe_allow_html=True)

    df = pd.DataFrame([
        {
            "Model": model,
            **metrics
        }
        for model, metrics in model_metrics.items()
    ])

    st.dataframe(df)

    st.write("### 📌 Accuracy Comparison")
    st.plotly_chart(px.bar(df, x="Model", y="Accuracy", color="Accuracy"))

    st.write("### 📌 ROC-AUC Comparison")
    st.plotly_chart(px.bar(df, x="Model", y="ROC_AUC", color="ROC_AUC"))

    st.write("### 📌 Radar Chart Summary")
    fig = go.Figure()
    for model, metrics in model_metrics.items():
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill="toself",
            name=model
        ))
    st.plotly_chart(fig)

    # 3D Visualization
    st.write("### 🌐 3D Feature Visualization")
    df_heart = pd.read_csv("heart.csv")
    fig3d = px.scatter_3d(
        df_heart, x="Age", y="Cholesterol", z="MaxHR",
        color="HeartDisease", color_continuous_scale=["green", "red"], height=600
    )
    st.plotly_chart(fig3d)

# --------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------
elif menu == "ℹ️ About":
    st.markdown("<div class='big-title'>ℹ️ About This App</div>", unsafe_allow_html=True)
    st.write("""
        This Heart Disease Prediction App uses multiple ML models trained 
        on clinical patient data, including charts, dashboards, dark mode, 
        3D visualization and much more.
    """)
    st.markdown("### 👨‍💻 Created by **Surya Paladi**")
