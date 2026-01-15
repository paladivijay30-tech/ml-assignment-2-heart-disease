import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fpdf import FPDF

#------------------------------------------------------------
# Streamlit Page Setup
#------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide"
)

#------------------------------------------------------------
# Dark Mode Toggle
#------------------------------------------------------------
dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
        .main {
            background-color: #1a1a1a;
            color: white;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(to bottom right, #dbeafe, #ffffff);
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)

#------------------------------------------------------------
# Sidebar Navigation
#------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3917/3917037.png", width=90)
    st.title("📌 Menu")

    menu = st.radio("Navigation", ["Home", "Predict", "About"])

    st.markdown("---")
    st.info("Created by **Surya Paladi**", icon="💡")

if menu == "Home":
    st.title("🏠 Welcome to the Heart Disease Prediction App")
    st.write("Use the sidebar to navigate.")
    st.stop()

if menu == "About":
    st.title("ℹ️ About This App")
    st.write("""
    This Heart Disease Prediction App uses machine learning  
    models to estimate your chance of heart disease based on  
    clinical parameters.

    **Tech Used:**  
    ✔️ Streamlit  
    ✔️ Scikit-Learn  
    ✔️ XGBoost  
    ✔️ Python  
    """)
    st.stop()

#------------------------------------------------------------
# Load Model & Scaler
#------------------------------------------------------------
model = pickle.load(open("XGBoost.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

#------------------------------------------------------------
# Page Header
#------------------------------------------------------------
st.image("https://cdn-icons-png.flaticon.com/512/820/820535.png", width=100)
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to check heart disease risk.")

#------------------------------------------------------------
# Input Form (UI Section)
#------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 45)
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)

with col2:
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"])
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST"])
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    st_slope = st.selectbox("ST Slope", ["Flat", "Up"])

oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

#------------------------------------------------------------
# Prepare input dictionary to match model’s feature order
#------------------------------------------------------------
input_dict = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': 1 if fasting_bs == "Yes" else 0,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_M': 1 if sex == "M" else 0,
    'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
    'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
    'ChestPainType_TA': 1 if chest_pain == "TA" else 0,
    'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
    'RestingECG_ST': 1 if resting_ecg == "ST" else 0,
    'ExerciseAngina_Y': 1 if exercise_angina == "Y" else 0,
    'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
    'ST_Slope_Up': 1 if st_slope == "Up" else 0
}

required_order = [
    'Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak',
    'Sex_M','ChestPainType_ATA','ChestPainType_NAP','ChestPainType_TA',
    'RestingECG_Normal','RestingECG_ST','ExerciseAngina_Y',
    'ST_Slope_Flat','ST_Slope_Up'
]

input_df = pd.DataFrame([input_dict])[required_order]
input_scaled = scaler.transform(input_df)

#------------------------------------------------------------
# PDF Generation Function
#------------------------------------------------------------
def generate_pdf(result, data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, "Heart Disease Prediction Report", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", size=11)

    for k, v in data.items():
        pdf.cell(200, 8, f"{k}: {v}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Prediction: {result}", ln=True)

    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        return f.read()

#------------------------------------------------------------
# Prediction Button
#------------------------------------------------------------
if st.button("Predict"):
    output = model.predict(input_scaled)[0]

    if output == 1:
        st.error("⚠️ High Risk of Heart Disease", icon="❗")
        st.snow()
        result_text = "High Risk"
    else:
        st.success("💚 Low Risk of Heart Disease", icon="✅")
        st.balloons()
        result_text = "Low Risk"

    pdf_file = generate_pdf(result_text, input_dict)
    st.download_button("📄 Download Patient Report", pdf_file, "Heart_Report.pdf")

#------------------------------------------------------------
# Footer
#------------------------------------------------------------
st.markdown("""
<hr>
<div style='text-align: center;'>
    Developed by <b>Surya Paladi</b> | ❤️ Machine Learning Project
</div>
""", unsafe_allow_html=True)
