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

    st.subheader("Applicant Information")
    applicant_age = st.number_input("Applicant Age", min_value=18, max_value=99)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0)
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
    # use options from schema so they match training data
    employment_status = st.selectbox(
        "Employment Status",
        categorical_schema["employment_status"]
    )

    st.subheader("Loan Details")
    loan_amount_requested = st.number_input("Loan Amount Requested (₹)", min_value=0)
    loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1)
    interest_rate_offered = st.number_input("Interest Rate (%)", min_value=0.0)
    loan_type = st.selectbox(
        "Loan Type",
        categorical_schema["loan_type"]
    )

    st.subheader("Credit History")
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
    existing_emis_monthly = st.number_input("Existing EMIs (Monthly)", min_value=0)

    st.subheader("Transaction Behavior")
    txn_failed_count = st.number_input("Failed Transactions", min_value=0)
    txn_avg_amount = st.number_input("Average Transaction Amount (₹)", min_value=0)
    txn_total_amount = st.number_input("Total Transaction Amount (₹)", min_value=0)
    txn_avg_balance_after = st.number_input("Average Balance After Txn (₹)", min_value=0)
    txn_distinct_devices = st.number_input("Distinct Devices Used", min_value=0)
    txn_distinct_merchants = st.number_input("Distinct Merchants Used", min_value=0)

    submitted = st.form_submit_button("Predict Fraud")


if submitted:
    # --- 1) start with default values for ALL features the model expects ---
    input_data = {}

    # numeric features: default 0.0
    for col in numeric_features:
        input_data[col] = 0.0

    # categorical features: default to first category from schema
    for col, options in categorical_schema.items():
        input_data[col] = options[0] if options else None

    # --- 2) overwrite with values from the form (the important features) ---
    input_data.update({
        "applicant_age": applicant_age,
        "monthly_income": monthly_income,
        "number_of_dependents": number_of_dependents,
        "employment_status": employment_status,
        "loan_amount_requested": loan_amount_requested,
        "loan_tenure_months": loan_tenure_months,
        "interest_rate_offered": interest_rate_offered,
        "loan_type": loan_type,
        "cibil_score": cibil_score,
        "existing_emis_monthly": existing_emis_monthly,
        "txn_failed_count": txn_failed_count,
        "txn_avg_amount": txn_avg_amount,
        "txn_total_amount": txn_total_amount,
        "txn_avg_balance_after": txn_avg_balance_after,
        "txn_distinct_devices": txn_distinct_devices,
        "txn_distinct_merchants": txn_distinct_merchants,
    })

    # --- 3) build DataFrame for the model ---
    X_input = pd.DataFrame([input_data])

    # --- 4) predict ---
    prob_fraud = model.predict_proba(X_input)[0][1]
    is_fraud = prob_fraud >= threshold

    st.subheader("Prediction Result")
    st.metric("Fraud Probability", f"{prob_fraud * 100:.2f}%")

    if is_fraud:
        st.error("⚠️ Likely Fraudulent")
    else:
        st.success("✅ Likely Legitimate")

    st.caption(f"Decision threshold: {threshold:.2f}")
