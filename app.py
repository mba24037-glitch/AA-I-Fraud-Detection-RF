import streamlit as st
import pandas as pd
import joblib
import json

# ----- Load model bundle -----
bundle = joblib.load("model.pkl")


model = bundle["model"]
threshold = bundle["threshold"]

# ----- Load feature schema -----
with open("feature_columns.json", "r") as f:
    schema = json.load(f)

numeric_features = schema["numeric"]
categorical_schema = schema["categorical"]

st.set_page_config(page_title="Loan Fraud Detection", page_icon="🕵️", layout="centered")

st.title("🕵️ Loan Fraud Detection")
st.write("Predict likelihood of fraud for a loan application.")

st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Numeric Inputs")

    num_inputs = {}
    for col in numeric_features:
        label = col.replace("_", " ").title()
        num_inputs[col] = st.number_input(label, value=0.0)

    st.subheader("Categorical Inputs")
    
    cat_inputs = {}
    for col, options in categorical_schema.items():
        label = col.replace("_", " ").title()
        cat_inputs[col] = st.selectbox(label, options)

    submitted = st.form_submit_button("Predict")

if submitted:
    X_input = pd.DataFrame([{**num_inputs, **cat_inputs}])
    prob_fraud = model.predict_proba(X_input)[0][1]
    is_fraud = prob_fraud >= threshold

    st.subheader("Prediction Result")
    st.metric("Fraud Probability", f"{prob_fraud*100:.2f}%")

    if is_fraud:
        st.error("⚠️ Likely Fraudulent")
    else:
        st.success("✅ Likely Legitimate")

    st.caption(f"Decision threshold: {threshold:.2f}")
