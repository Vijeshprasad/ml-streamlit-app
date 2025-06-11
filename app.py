import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset (change path as needed)
df = pd.read_csv("autoinsurance.csv")

# ---- Select the 10 features ----
features = [
    "Age",
    "Gender",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
    "Region_Code",
    "Driving_License",
    "Previously_Insured"
]

# Encode categorical features to match app
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Vehicle_Damage"] = df["Vehicle_Damage"].map({"Yes": 1, "No": 0})
df["Vehicle_Age"] = df["Vehicle_Age"].map({
    "> 2 Years": 2,
    "1-2 Year": 1,
    "< 1 Year": 0
})

# Input features and target
X = df[features]
y = df["Response"]  # Or whatever your target column is

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Fit and save StandardScaler ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler.pkl")

# ---- Train and save model ----
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)
joblib.dump(model, "model.pkl")

print("✅ Model and scaler saved.")
