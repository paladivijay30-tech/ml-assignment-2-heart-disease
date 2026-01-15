import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
from io import BytesIO

# -----------------------------------------------------
# CUSTOM CSS (Dynamic Color Themes)
# -----------------------------------------------------

def apply_dark_mode(dark):
    if dark:
        css = """
        <style>
        body, .stApp { background-color: #0E1117 !important; color: white !important; }
        .sidebar .sidebar-content { background-color: #1A1D23 !important; }
        .stButton>button { background-color: #4A4A4A !important; color: white !important; }
        .stSelectbox div, .stTextInput>div>div input { color: white !important; }
        </style>
        """
    else:
        css = """
        <style>
        body, .stApp { background-color: #F7F7F7 !important; color: black !important; }
        .sidebar .sidebar-content { background-color: #ECECEC !important; }
        .stButton>button { background-color: #FFFFFF !important; color: black !important; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# -----------------------------------------------------
# PDF GENERATION (RGB Beautiful Medical Report)
# -----------------------------------------------------

def generate_pdf(input_data, prediction_label):

    pdf = FPDF()
    pdf.add_page()

    # Colors
    title_color = (0, 102, 204)
    header_color = (50, 50, 50)
    good = (0, 170, 0)
    bad = (200, 0, 0)

    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(*title_color)
    pdf.cell(0, 12, "Heart Disease Prediction Report", ln=True, align="C")

    pdf.ln(8)

    # Patient Data
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(*header_color)
    pdf.cell(0, 10, "Patient Information:", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)

    for key, value in input_data.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)

    pdf.ln(10)

    # Prediction Section
    pdf.set_font("Arial", "B", 16)

    if prediction_label == "High Risk":
        pdf.set_text_color(*bad)
    else:
        pdf.set_text_color(*good)

    pdf.cell(0, 12, f"Prediction: {prediction_label}", ln=True)

    # Export PDF
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# -----------------------------------------------------
# MAIN APP
# -----------------------------------------------------
st.set_page_config(page_title="Heart Disease App", layout="wide")

# DARK MODE TOGGLE
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
apply_dark_mode(dark_mode)

st.title("❤️ Heart Disease Prediction App")

# INPUT FORM
with st.form("form"):

    age = st.number_input("Age", 1, 120)
    resting_bp = st.number_input("Resting BP", 50, 200)
    chol = st.number_input("Cholesterol", 50, 400)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    max_hr = st.number_input("Max Heart Rate", 60, 220)
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0)

    sex = st.selectbox("Sex", ["M", "F"])
    chest = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST"])
    angina = st.selectbox("Exercise Angina", ["Y", "N"])
    slope = st.selectbox("ST Slope", ["Up", "Flat"])

    submit = st.form_submit_button("Predict")

# -----------------------------------------------------
# PROCESS INPUT + MODEL PREDICTION
# -----------------------------------------------------
if submit:

    # Dummy prediction (replace with your MODEL)
    pred = np.random.choice([0, 1])
    label = "High Risk" if pred == 1 else "Low Risk"

    if pred == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # Create dictionary for PDF
    input_dict = {
        "Age": age, "RestingBP": resting_bp, "Cholesterol": chol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr, "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ChestPainType_ATA": 1 if chest == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest == "NAP" else 0,
        "ChestPainType_TA": 1 if chest == "TA" else 0,
        "RestingECG_Normal": 1 if rest_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if rest_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if angina == "Y" else 0,
        "ST_Slope_Up": 1 if slope == "Up" else 0,
        "ST_Slope_Flat": 1 if slope == "Flat" else 0,
    }

    pdf = generate_pdf(input_dict, label)

    st.download_button(
        label="📄 Download Patient Report (PDF)",
        data=pdf,
        file_name="Heart_Report.pdf",
        mime="application/pdf",
    )
