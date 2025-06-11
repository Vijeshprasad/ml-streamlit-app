import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("insurance_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚗 Auto Insurance Claim Prediction")

st.write("Enter the customer details below to predict the likelihood of making a claim:")

# Input fields matching your dataset's features
age = st.slider("Age", 18, 100, 30)
driving_license = st.selectbox("Has Driving License?", [0, 1])
previously_insured = st.selectbox("Previously Insured?", [0, 1])
annual_premium = st.number_input("Annual Premium", 1000, 100000, 30000)
policy_sales_channel = st.number_input("Policy Sales Channel", 0, 200, 26)
vintage = st.slider("Days Since Policy Initiated (Vintage)", 0, 300, 150)

vehicle_damage = st.selectbox("Vehicle Damaged Before?", ["No", "Yes"])
gender = st.selectbox("Gender", ["Male", "Female"])
vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])

# Process categorical variables
vehicle_damage = 1 if vehicle_damage == "Yes" else 0
gender_male = 1 if gender == "Male" else 0
vehicle_age_1_2 = 1 if vehicle_age == "1-2 Year" else 0
vehicle_age_gt_2 = 1 if vehicle_age == "> 2 Years" else 0

# Prepare input for prediction (order matters)
input_data = np.array([[age, driving_license, previously_insured, annual_premium,
                        policy_sales_channel, vintage, vehicle_damage,
                        gender_male, vehicle_age_1_2, vehicle_age_gt_2]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Claim Probability"):
    probability = model.predict_proba(input_scaled)[0][1]
    st.success(f"📊 Predicted Claim Probability: {probability * 100:.2f}%")
