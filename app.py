import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and configuration
model = joblib.load("models/final_catboost_model.pkl")
config = joblib.load("models/final_model_config.pkl")

threshold = config["threshold"]

st.title("Credit Card Fraud Detection")

st.write("Enter transaction details below.")

# Feature inputs
features = [
    "V17", "V14", "V10", "V12", "V11",
    "V16", "V3", "V4", "V9", "V18",
    "V7", "V2", "V27", "V21", "V6",
    "V5", "V1", "V28", "V8", "LogAmount"
]

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        feature,
        value=0.0
    )

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    probability = model.predict_proba(input_df)[0, 1]

    if probability >= threshold:
        st.error("⚠️ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: {probability:.4f}")