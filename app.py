import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load pre-trained model and scaler (uncomment when you have these files)
# model = joblib.load("insurance_model.pkl")
# scaler = joblib.load("scaler.pkl")

st.title("Auto Insurance Claim Prediction")

# Initialize session state for storing predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

st.header("Customer Information")
st.subheader("Demographics")

# Section 1: Customer Demographics
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired", "Medical Leave", "Disabled"])
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    
with col2:
    education = st.selectbox("Education Level", ["High School or Below", "College", "Bachelor", "Master", "Doctor"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    occupation = st.selectbox("Occupation", ["Professional", "Laborer", "Skilled Manual", "Clerical", "Housewife", "Student", "Retired", "Unemployed"])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    years_at_residence = st.number_input("Years at Current Residence", min_value=0, max_value=50, value=5)

# Section 2: Policy Information
st.subheader("Policy Details")
policy_col1, policy_col2 = st.columns(2)
with policy_col1:
    coverage = st.selectbox("Coverage Type", ["Basic", "Extended", "Premium"])
    policy_type = st.selectbox("Policy Type", ["Personal Auto", "Corporate Auto", "Special Auto"])
    renew_offer = st.selectbox("Renewal Offer Type", ["1", "2", "3", "4"])
    policy_term = st.selectbox("Policy Term", ["Annual", "Semi-Annual", "Monthly"])
    deductible = st.number_input("Deductible Amount ($)", min_value=0, value=500)
    
with policy_col2:
    sales_channel = st.selectbox("Sales Channel", ["Web", "Branch", "Agent", "Call Center"])
    months_since_policy_inception = st.number_input("Months Since Policy Inception", min_value=0, value=12)
    number_of_policies = st.number_input("Number of Policies", min_value=1, value=1)
    prior_insurance = st.selectbox("Prior Insurance Coverage", ["None", "Some", "Continuous"])
    loyalty_discount = st.selectbox("Loyalty Discount", ["None", "Bronze", "Silver", "Gold"])

# Section 3: Vehicle Information
st.subheader("Vehicle Information")
vehicle_col1, vehicle_col2 = st.columns(2)
with vehicle_col1:
    vehicle_class = st.selectbox("Vehicle Class", ["Two-Door Car", "Four-Door Car", "SUV", "Sports Car", "Luxury Car", "Luxury SUV"])
    vehicle_size = st.selectbox("Vehicle Size", ["Small", "Medsize", "Large"])
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
    vehicle_value = st.number_input("Vehicle Value ($)", min_value=0, value=25000)
    anti_lock_brakes = st.selectbox("Anti-Lock Brakes", ["Yes", "No"])
    
with vehicle_col2:
    monthly_premium = st.number_input("Monthly Premium ($)", min_value=0, value=100)
    total_claim_amount = st.number_input("Total Claim Amount ($)", min_value=0.0, value=0.0)
    annual_mileage = st.number_input("Annual Mileage", min_value=0, value=12000)
    safety_features = st.number_input("Number of Safety Features", min_value=0, max_value=10, value=3)
    garaged = st.selectbox("Vehicle Garaged", ["Yes", "No"])

# Section 4: Claim History
st.subheader("Claim History")
claim_col1, claim_col2 = st.columns(2)
with claim_col1:
    months_since_last_claim = st.number_input("Months Since Last Claim", min_value=0, value=12)
    number_of_open_complaints = st.number_input("Number of Open Complaints", min_value=0, value=0)
    past_claims = st.number_input("Number of Past Claims", min_value=0, value=0)
    claim_severity = st.selectbox("Historical Claim Severity", ["None", "Low", "Medium", "High"])
    
with claim_col2:
    customer_lifetime_value = st.number_input("Customer Lifetime Value ($)", min_value=0, value=10000)
    state = st.selectbox("State", ["California", "Washington", "Oregon", "Arizona", "Nevada"])
    fraud_reports = st.number_input("Number of Fraud Reports", min_value=0, value=0)
    claim_frequency = st.selectbox("Claim Frequency", ["None", "Rare", "Occasional", "Frequent"])

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
occupation_map = {
    "Professional": 0,
    "Laborer": 1,
    "Skilled Manual": 2,
    "Clerical": 3,
    "Housewife": 4,
    "Student": 5,
    "Retired": 6,
    "Unemployed": 7
}
policy_term_map = {"Annual": 0, "Semi-Annual": 1, "Monthly": 2}
prior_insurance_map = {"None": 0, "Some": 1, "Continuous": 2}
loyalty_discount_map = {"None": 0, "Bronze": 1, "Silver": 2, "Gold": 3}
claim_severity_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
claim_frequency_map = {"None": 0, "Rare": 1, "Occasional": 2, "Frequent": 3}
yes_no_map = {"Yes": 1, "No": 0}

# Create input array with all 33 features in the correct order
input_data = [
    age,
    gender_map[gender],
    marital_map[marital_status],
    education_map[education],
    employment_map[employment_status],
    occupation_map[occupation],
    income,
    location_map[location],
    credit_score,
    years_at_residence,
    coverage_map[coverage],
    policy_map[policy_type],
    renew_offer,
    policy_term_map[policy_term],
    deductible,
    sales_map[sales_channel],
    months_since_policy_inception,
    number_of_policies,
    prior_insurance_map[prior_insurance],
    loyalty_discount_map[loyalty_discount],
    vehicle_class_map[vehicle_class],
    vehicle_size_map[vehicle_size],
    vehicle_age,
    vehicle_value,
    anti_lock_brakes,
    monthly_premium,
    total_claim_amount,
    annual_mileage,
    safety_features,
    garaged,
    months_since_last_claim,
    number_of_open_complaints,
    past_claims,
    claim_severity_map[claim_severity],
    customer_lifetime_value,
    state_map[state],
    fraud_reports,
    claim_frequency_map[claim_frequency]
]

# Keep only the first 33 features (adjust based on your actual model)
input_data = input_data[:33]

# Prediction button
if st.button("Predict Claim Likelihood"):
    try:
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # For demo purposes - generate random probabilities
        # Replace this with actual model prediction when ready
        if 'model' in globals():
            input_scaled = scaler.transform(input_array)
            prediction_proba = model.predict_proba(input_scaled)[0][1]
            prediction = 1 if prediction_proba >= 0.5 else 0
        else:
            # Demo mode - random probability between 0.1 and 0.9 based on inputs
            risk_score = sum(input_data) / (len(input_data) * max(input_data)) if max(input_data) > 0 else 0.5
            prediction_proba = min(max(0.1, risk_score * 1.5), 0.9)
            prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Store prediction in session state
        st.session_state.predictions.append({
            "features": input_data.copy(),
            "probability": prediction_proba,
            "prediction": prediction
        })
        
        # Display results
        if prediction == 1:
            st.error(f"🚨 High claim risk predicted ({prediction_proba*100:.1f}% probability)")
            st.markdown("""
            **Recommended actions:**
            - Review policy details
            - Consider higher deductible
            - Schedule customer review
            - Flag for additional verification
            """)
        else:
            st.success(f"✅ Low claim risk predicted ({100-prediction_proba*100:.1f}% probability)")
            st.markdown("""
            **Customer appears low-risk**
            - Eligible for standard rates
            - Consider loyalty rewards
            """)
            
        # Show more detailed output
        with st.expander("Detailed Prediction Information"):
            st.write("### Input Features Summary")
            feature_df = pd.DataFrame({
                "Feature": [
                    "Age", "Gender", "Marital Status", "Education", "Employment Status",
                    "Occupation", "Income", "Location", "Credit Score", "Years at Residence",
                    "Coverage Type", "Policy Type", "Renewal Offer", "Policy Term", "Deductible",
                    "Sales Channel", "Months Since Inception", "Number of Policies",
                    "Prior Insurance", "Loyalty Discount", "Vehicle Class", "Vehicle Size",
                    "Vehicle Age", "Vehicle Value", "Anti-Lock Brakes", "Monthly Premium",
                    "Total Claim Amount", "Annual Mileage", "Safety Features", "Garaged",
                    "Months Since Last Claim", "Open Complaints", "Past Claims"
                ],
                "Value": input_data
            })
            st.dataframe(feature_df)
            
            st.write(f"### Prediction Confidence: {prediction_proba*100:.1f}%")
            st.progress(prediction_proba)
            
            if len(st.session_state.predictions) > 1:
                st.write("### Previous Predictions")
                prev_df = pd.DataFrame({
                    "Prediction": ["High Risk" if p["prediction"] == 1 else "Low Risk" for p in st.session_state.predictions[:-1]],
                    "Probability": [p["probability"] for p in st.session_state.predictions[:-1]]
                })
                st.dataframe(prev_df)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please ensure all input fields are filled correctly.")

# Add some explanatory text
st.markdown("""
### About This Prediction
- **High Risk (≥50% probability):** Customer has elevated likelihood of filing a claim
- **Low Risk (<50% probability):** Customer has normal risk profile

**Note:** For actual deployment:
1. Train a model on your historical data
2. Save the model and scaler as .pkl files
3. Uncomment the model loading code at the top
""")

# Add some styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .st-b7 {
        color: white;
    }
    .st-c0 {
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)
