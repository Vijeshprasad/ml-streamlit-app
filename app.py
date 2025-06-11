import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("ML Prediction App (10 Features)")

# Create input fields for 10 features
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)
feature_4 = st.number_input("Feature 4", value=0.0)
feature_5 = st.number_input("Feature 5", value=0.0)
feature_6 = st.number_input("Feature 6", value=0.0)
feature_7 = st.number_input("Feature 7", value=0.0)
feature_8 = st.number_input("Feature 8", value=0.0)
feature_9 = st.number_input("Feature 9", value=0.0)
feature_10 = st.number_input("Feature 10", value=0.0)

# Collect inputs
input_data = np.array([
    feature_1, feature_2, feature_3, feature_4, feature_5,
    feature_6, feature_7, feature_8, feature_9, feature_10
]).reshape(1, -1)

# Make prediction
if st.button("Predict"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
