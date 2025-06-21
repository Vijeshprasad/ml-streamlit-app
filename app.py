import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Auto Insurance Claim Prediction")

st.header("Customer Information")
st.subheader("Demographics")

# Section 1: Customer Demographics
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired", "Medical Leave", "Disabled"])

with col2:
    education = st.selectbox("Education Level", ["High School or Below", "College", "Bachelor", "Master", "Doctor"])
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])

# Section 2: Policy Information
st.subheader("Policy Details")
policy_col1, policy_col2 = st.columns(2)
with policy_col1:
    coverage = st.selectbox("Coverage Type", ["Basic", "Extended", "Premium"])
    policy_type = st.selectbox("Policy Type", ["Personal Auto", "Corporate Auto", "Special Auto"])
    renew_offer = st.selectbox("Renewal Offer Type", ["1", "2", "3", "4"])

with policy_col2:
    sales_channel = st.selectbox("Sales Channel", ["Web", "Branch", "Agent", "Call Center"])
    months_since_policy_inception = st.number_input("Months Since Policy Inception", min_value=0, value=12)
    number_of_policies = st.number_input("Number of Policies", min_value=1, value=1)

# Section 3: Vehicle Information
st.subheader("Vehicle Information")
vehicle_col1, vehicle_col2 = st.columns(2)
with vehicle_col1:
    vehicle_class = st.selectbox("Vehicle Class", ["Two-Door Car", "Four-Door Car", "SUV", "Sports Car", "Luxury Car", "Luxury SUV"])
    vehicle_size = st.selectbox("Vehicle Size", ["Small", "Medsize", "Large"])

with vehicle_col2:
    monthly_premium = st.number_input("Monthly Premium ($)", min_value=0, value=100)
    total_claim_amount = st.number_input("Total Claim Amount ($)", min_value=0.0, value=0.0)

# Section 4: Claim History
st.subheader("Claim History")
claim_col1, claim_col2 = st.columns(2)
with claim_col1:
    months_since_last_claim = st.number_input("Months Since Last Claim", min_value=0, value=12)
    number_of_open_complaints = st.number_input("Number of Open Complaints", min_value=0, value=0)

with claim_col2:
    customer_lifetime_value = st.number_input("Customer Lifetime Value ($)", min_value=0, value=10000)
    state = st.selectbox("State", ["California", "Washington", "Oregon", "Arizona", "Nevada"])

# Create feature mapping dictionaries
gender_map = {"Male": 0, "Female": 1}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
employment_map = {
    "Employed": 1, 
    "Unemployed": 0, 
    "Retired": 3, 
    "Medical Leave": 2, 
    "Disabled": 4
}
education_map = {
    "High School or Below": 0,
    "College": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctor": 4
}
location_map = {"Urban": 2, "Suburban": 1, "Rural": 0}
coverage_map = {"Basic": 0, "Extended": 1, "Premium": 2}
policy_map = {"Personal Auto": 0, "Corporate Auto": 1, "Special Auto": 2}
sales_map = {"Web": 0, "Branch": 1, "Agent": 2, "Call Center": 3}
vehicle_class_map = {
    "Two-Door Car": 0,
    "Four-Door Car": 1,
    "SUV": 4,
    "Sports Car": 2,
    "Luxury Car": 3,
    "Luxury SUV": 5
}
vehicle_size_map = {"Small": 0, "Medsize": 1, "Large": 2}
state_map = {
    "California": 0,
    "Washington": 1,
    "Oregon": 2,
    "Arizona": 3,
    "Nevada": 4
}

# Create input array
input_data = [
    customer_lifetime_value,
    coverage_map[coverage],
    education_map[education],
    employment_map[employment_status],
    gender_map[gender],
    income,
    location_map[location],
    marital_map[marital_status],
    monthly_premium,
    months_since_last_claim,
    months_since_policy_inception,
    number_of_open_complaints,
    number_of_policies,
    policy_map[policy_type],
    int(renew_offer),
    sales_map[sales_channel],
    total_claim_amount,
    vehicle_class_map[vehicle_class],
    vehicle_size_map[vehicle_size],
    state_map[state]
]

# Extend with additional features if used during model training
input_data.extend([
    coverage_map[coverage],
    education_map[education],
    employment_map[employment_status],
    location_map[location],
    marital_map[marital_status],
    policy_map[policy_type],
])

# Prediction button
if st.button("Predict Claim Likelihood"):
    try:
        # Convert and scale input
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Actual prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        # Display prediction
        if prediction == 1:
            st.error(f"High claim risk predicted ({prediction_proba*100:.1f}% probability)")
            st.write("Recommended actions: Review policy, consider risk mitigation strategies")
        else:
            st.success(f"Low claim risk predicted ({100 - prediction_proba*100:.1f}% probability)")
            st.write("Customer appears low-risk")

        # Expand for details
        with st.expander("Detailed Prediction Information"):
            st.write("Raw input features:", input_data)
            st.write("Scaled features:", input_scaled.tolist()[0])
            st.write(f"Prediction confidence: {prediction_proba*100:.1f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

st.markdown("""
**Note:** This is a working insurance claim prediction app using a trained model.
Ensure the feature order and transformations match your training pipeline.
""")
