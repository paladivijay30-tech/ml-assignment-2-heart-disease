import streamlit as st
import pandas as pd
import pickle

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered"
)

# ===============================
# CUSTOM CSS FOR BEAUTIFUL UI
# ===============================
st.markdown("""
    <style>
        .main { background-color: #f7f9fc; }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            height: 45px;
            width: 160px;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #ff1e1e;
            color: white;
        }
        .info-card {
            padding: 20px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = pickle.load(open("Random_Forest.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Required training columns (exact order)
required_cols = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_M',
    'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_Y',
    'ST_Slope_Flat', 'ST_Slope_Up'
]

# ===============================
# PAGE TITLE
# ===============================
st.markdown("<h1 style='text-align:center;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Enter patient details below to analyze the risk of heart disease.</p>", unsafe_allow_html=True)


# ===============================
# FORM CARD
# ===============================
with st.container():
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)

    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["M", "F"])
        resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
        max_hr = st.number_input("Max Heart Rate", 50, 250, 150)
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)

    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST"])
    exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
    st_slope = st.selectbox("ST Slope", ["Flat", "Up"])

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# BUILD INPUT DICT
# ===============================
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

# Convert to correct DataFrame
input_df = pd.DataFrame([input_dict])[required_cols]

# ===============================
# PREDICT BUTTON
# ===============================
st.markdown("<div class='info-card'>", unsafe_allow_html=True)
st.subheader("Prediction")

if st.button("Predict"):
    scaled = scaler.transform(input_df)
    output = model.predict(scaled)[0]

    if output == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

st.markdown("</div>", unsafe_allow_html=True)
