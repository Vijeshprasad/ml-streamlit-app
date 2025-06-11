import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Auto Insurance Claim Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
vehicle_age = st.selectbox("Vehicle Age", ["> 2 Years", "1-2 Year", "< 1 Year"])
vehicle_damage = st.selectbox("Vehicle Damage", ["Yes", "No"])
annual_premium = st.number_input("Annual Premium", value=30000)
policy_sales_channel = st.number_input("Policy Sales Channel", value=26)
vintage = st.number_input("Vintage (Customer since days)", value=150)
region_code = st.number_input("Region Code", value=28)
driving_license = st.radio("Driving License", [1, 0])
previously_insured = st.radio("Previously Insured", [1, 0])

# Encoding categorical values (same as during training)
gender_encoded = 1 if gender == "Male" else 0
vehicle_age_encoded = {
    "> 2 Years": 2,
    "1-2 Year": 1,
    "< 1 Year": 0
}[vehicle_age]
vehicle_damage_encoded = 1 if vehicle_damage == "Yes" else 0

# Create feature array
input_data = np.array([
    age,
    gender_encoded,
    vehicle_age_encoded,
    vehicle_damage_encoded,
    annual_premium,
    policy_sales_channel,
    vintage,
    region_code,
    driving_license,
    previously_insured
]).reshape(1, -1)

# Predict
if st.button("Predict Claim"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = "Will File a Claim" if prediction[0] == 1 else "No Claim"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error: {e}")
