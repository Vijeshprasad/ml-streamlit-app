# Prediction button
if st.button("Predict Claim Likelihood"):
    try:
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # For demo purposes - generate varied probabilities based on inputs
        # Replace this with actual model prediction when ready
        if 'model' in globals():
            input_scaled = scaler.transform(input_array)
            prediction_proba = model.predict_proba(input_scaled)[0][1]
            prediction = 1 if prediction_proba >= 0.5 else 0
        else:
            # DEMO MODE - Improved risk calculation
            # Define risk factors and their weights
            risk_factors = {
                'age': (age, 0.1, lambda x: x/100),  # Older drivers slightly higher risk
                'income': (income, 0.05, lambda x: 1 - min(x/200000, 1)),  # Higher income = lower risk
                'credit_score': (credit_score, 0.1, lambda x: 1 - (x-300)/550),  # Lower credit = higher risk
                'months_since_last_claim': (months_since_last_claim, 0.15, lambda x: 1/(x+1)),  # Recent claims = higher risk
                'past_claims': (past_claims, 0.2, lambda x: min(x/5, 1)),  # More past claims = higher risk
                'vehicle_age': (vehicle_age, 0.1, lambda x: min(x/20, 1)),  # Older cars = higher risk
                'annual_mileage': (annual_mileage, 0.1, lambda x: min(x/30000, 1)),  # More miles = higher risk
                'fraud_reports': (fraud_reports, 0.2, lambda x: min(x*0.5, 1))  # Any fraud reports = much higher risk
            }
            
            # Calculate weighted risk score
            total_weight = sum([weight for _, (_, weight, _) in risk_factors.items()])
            weighted_risk = sum([value * weight * transform(value) 
                               for name, (value, weight, transform) in risk_factors.items()])
            
            # Normalize to 0.1-0.9 range
            base_risk = weighted_risk / total_weight
            prediction_proba = min(max(0.1, base_risk * 1.3), 0.9)  # Keep within reasonable bounds
            prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Store prediction in session state
        st.session_state.predictions.append({
            "features": input_data.copy(),
            "probability": prediction_proba,
            "prediction": prediction
        })
        
        # Display results (rest of your display code remains the same)
