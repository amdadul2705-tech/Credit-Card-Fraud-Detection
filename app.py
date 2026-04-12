import streamlit as st
import numpy as np
import joblib

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Fraud Detection App")

amount = st.number_input("Transaction Amount", value=100.0)

# Create feature array matching the 30 features from training
features = np.zeros((1, 30))
# Scale the amount and place in the last position (Amount was last feature)
features[0][29] = scaler.transform([[amount]])[0][0]

if st.button("Check Transaction"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legit Transaction (Probability: {prob:.2f})")