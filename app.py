import streamlit as st
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction App",
    layout="wide",
    page_icon="❤️"
)

# --------------------------------------------------------
# CUSTOM CSS FOR DARK/LIGHT MODE
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
        .section {
            background-color: rgba(255,255,255,0.07);
            padding: 1.3rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --------------------------------------------------------
# DARK MODE TOGGLE
# --------------------------------------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
            body { background-color: #0E1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------------
# SIDEBAR MENU
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
# PREDEFINED MODEL METRICS (OPTION A)
# --------------------------------------------------------
model_metrics = {
    "Logistic Regression": {"Accuracy": 0.84, "AUC": 0.88},
    "Decision Tree": {"Accuracy": 0.78, "AUC": 0.75},
    "Random Forest": {"Accuracy": 0.91, "AUC": 0.93},
    "XGBoost": {"Accuracy": 0.92, "AUC": 0.94},
    "KNN": {"Accuracy": 0.82, "AUC": 0.80},
    "Naive Bayes": {"Accuracy": 0.79, "AUC": 0.77},
}

# --------------------------------------------------------
# INPUT FORM FUNCTION
# --------------------------------------------------------
def get_user_input():
    st.markdown("<div class='sub-title'>Enter Patient Details</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        resting_bp = st.number_input("Resting BP", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
        max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)

    with col2:
        sex = st.selectbox("Sex", ["M", "F"])
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST"])
        exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
        st_slope = st.selectbox("ST Slope", ["Flat", "Up"])

    input_dict = {
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

    return input_dict


# --------------------------------------------------------
# PREDICT PAGE
# --------------------------------------------------------
if menu == "🔮 Predict":
    st.markdown("<div class='big-title'>❤️ Heart Disease Prediction</div>", unsafe_allow_html=True)

    input_dict = get_user_input()
    input_df = pd.DataFrame([input_dict])

    # Scale
    scaled = scaler.transform(input_df)

    # Prediction using best model (XGBoost)
    best_model = models["XGBoost"]
    pred = best_model.predict(scaled)[0]

    if st.button("Predict"):
        if pred == 1:
            st.error("⚠️ High Risk of Heart Disease", icon="🚨")
        else:
            st.success("✅ Low Risk of Heart Disease", icon="💚")


# --------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------
elif menu == "🏠 Home":
    st.markdown("<div class='big-title'>❤️ Heart Disease Prediction App</div>", unsafe_allow_html=True)
    st.write("This application predicts heart disease using multiple ML models.")


def render_3d_plot(df):
    st.markdown("### 🌐 3D Feature Visualization")

    fig = px.scatter_3d(
        df,
        x="Age",
        y="Cholesterol",
        z="MaxHR",
        color="HeartDisease",
        symbol="HeartDisease",
        color_continuous_scale=["green", "red"],
        title="3D Scatter Plot: Age vs Cholesterol vs MaxHR",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------
# MODEL COMPARISON PAGE
# --------------------------------------------------------
elif menu == "📊 Model Comparison":

    st.markdown("<div class='big-title'>📊 Model Comparison Dashboard</div>", unsafe_allow_html=True)

    df = pd.DataFrame([
        {"Model": m, "Accuracy": model_metrics[m]["Accuracy"], "AUC": model_metrics[m]["AUC"]}
        for m in model_metrics
    ])

    st.write("### 📌 Accuracy Comparison")
    fig1 = px.bar(df, x="Model", y="Accuracy", color="Accuracy")
    st.plotly_chart(fig1, use_container_width=True)

    st.write("### 📌 AUC Comparison")
    fig2 = px.bar(df, x="Model", y="AUC", color="AUC")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("### 📌 Radar Chart Comparison")
    fig = go.Figure()

    for m in model_metrics:
        fig.add_trace(go.Scatterpolar(
            r=[model_metrics[m]["Accuracy"], model_metrics[m]["AUC"]],
            theta=["Accuracy", "AUC"],
            fill='toself',
            name=m
        ))

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 🚀 ADD THIS: 3D Visualization
    # -----------------------------
    st.write("### 🌐 3D Feature Visualization")
    render_3d_plot(full_dataset_df) 



# --------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------
elif menu == "ℹ️ About":
    st.markdown("<div class='big-title'>ℹ️ About This App</div>", unsafe_allow_html=True)
    st.write("""
        This Heart Disease Prediction App uses multiple machine learning models trained 
        on clinical patient data. It includes charts, comparison dashboards, dark mode, 
        and a clean user interface.
    """)

    st.markdown("### 👨‍💻 Created by **Surya Paladi**")


