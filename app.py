import streamlit as st
import pickle
import pandas as pd

# Load the trained model & scaler
model = pickle.load(open("Random_Forest.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

# Collect inputs
age = st.number_input("Age", 20, 100, 45)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
cholesterol = st.number_input("Cholesterol", 100, 400, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST"])
max_hr = st.number_input("Max Heart Rate", 50, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", -5.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat"])

# Build input dictionary
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
    'ST_Slope_Up': 1 if st_slope == "Up" else 0,
}

input_df = pd.DataFrame([input_dict])

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    if pred == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
