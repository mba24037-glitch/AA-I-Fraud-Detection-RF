import streamlit as st
import pandas as pd
import joblib
import json

# =========================
#  Load model + schema
# =========================
bundle = joblib.load("model.pkl")
model = bundle["model"]
threshold = bundle["threshold"]

with open("feature_columns.json", "r") as f:
    schema = json.load(f)

numeric_features = schema["numeric"]
categorical_schema = schema["categorical"]

# =========================
#  Page config
# =========================
st.set_page_config(
    page_title="Loan Fraud Detection",
    page_icon="🕵️",
    layout="wide"
)

# =========================
#  Sidebar
# =========================
st.sidebar.title("🕵️ Loan Fraud Detection")
st.sidebar.markdown(
    """
This tool estimates the **probability of fraud** for a loan application  
using a **Random Forest model** trained on historical applications and
transaction behaviour.

**How to use:**
1. Fill in the key details on the right  
2. Click **Predict Fraud**  
3. Review the risk level and probability
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Legend**")
st.sidebar.markdown("🟢 **Low**: p < 20%")
st.sidebar.markdown("🟠 **Medium**: 20% ≤ p < 50%")
st.sidebar.markdown("🔴 **High**: p ≥ 50%")

st.sidebar.markdown("---")
st.sidebar.caption("Academic project – not for real credit decisions.")

# =========================
#  Header
# =========================
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0.2rem;">Loan Fraud Detection Dashboard</h1>
    <p style="text-align:center; color:gray; font-size:0.95rem;">
    Predicts the likelihood that a loan application is fraudulent based on applicant profile,
    loan details and recent transaction behaviour.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================
#  Input Form (Professional Layout)
# =========================
with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # ------- Applicant Information -------
    with col1:
        st.markdown("### 👤 Applicant")
        applicant_age = st.number_input("Age", min_value=18, max_value=99, value=30)
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=50000)
        number_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=0
        )
        employment_status = st.selectbox(
            "Employment Status",
            categorical_schema.get("employment_status", ["Salaried", "Self.Employed", "Unemployed"]),
        )

    # ------- Loan Details / Credit -------
    with col2:
        st.markdown("### 💰 Loan & Credit")
        loan_amount_requested = st.number_input(
            "Loan Amount Requested (₹)", min_value=0, value=500000
        )
        loan_tenure_months = st.number_input(
            "Tenure (Months)", min_value=1, value=60
        )
        interest_rate_offered = st.number_input(
            "Interest Rate (%)", min_value=0.0, value=12.0
        )
        loan_type = st.selectbox(
            "Loan Type",
            categorical_schema.get("loan_type", ["Home.Loan", "Car.Loan", "Personal.Loan"]),
        )
        cibil_score = st.number_input(
            "CIBIL Score", min_value=300, max_value=900, value=750
        )
        existing_emis_monthly = st.number_input(
            "Existing EMIs (Monthly, ₹)", min_value=0, value=0
        )

    # ------- Transaction Behaviour -------
    with col3:
        st.markdown("### 📊 Transactions (Past Behaviour)")
        txn_failed_count = st.number_input("Failed Txns (count)", min_value=0, value=0)
        txn_avg_amount = st.number_input(
            "Avg Txn Amount (₹)", min_value=0, value=2000
        )
        txn_total_amount = st.number_input(
            "Total Txn Amount (₹)", min_value=0, value=40000
        )
        txn_avg_balance_after = st.number_input(
            "Avg Balance After Txn (₹)", min_value=0, value=15000
        )
        txn_distinct_devices = st.number_input(
            "Distinct Devices Used", min_value=0, value=1
        )
        txn_distinct_merchants = st.number_input(
            "Distinct Merchants Used", min_value=0, value=3
        )

    st.markdown("")
    submit_col1, _, _ = st.columns([1, 3, 3])
    with submit_col1:
        submitted = st.form_submit_button("🔍 Predict Fraud", use_container_width=True)

# =========================
#  Prediction Logic
# =========================
if submitted:
    # 1) default values for every feature
    input_data = {}

    for col in numeric_features:
        input_data[col] = 0.0

    for col, options in categorical_schema.items():
        input_data[col] = options[0] if options else None

    # 2) overwrite with form values (our key variables)
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

    X_input = pd.DataFrame([input_data])

    # 3) predict
    prob_fraud = float(model.predict_proba(X_input)[0][1])
    is_fraud = prob_fraud >= threshold

    # =========================
    #  Result Display
    # =========================
    st.markdown("---")
    st.subheader("Prediction Summary")

    left, right = st.columns([1.2, 1])

    # ---- Left: main result ----
    with left:
        st.metric("Fraud Probability", f"{prob_fraud * 100:.2f}%")

        # risk label
        if prob_fraud < 0.20:
            risk_label = "Low Risk"
            emoji = "🟢"
        elif prob_fraud < 0.50:
            risk_label = "Medium Risk"
            emoji = "🟠"
        else:
            risk_label = "High Risk"
            emoji = "🔴"

        st.markdown(
            f"""
            <div style="
                padding: 0.8rem 1rem;
                border-radius: 0.5rem;
                background-color: #111827;
                border: 1px solid #374151;
                ">
                <span style="font-size:1.1rem;">{emoji} <b>{risk_label}</b></span><br>
                <span style="font-size:0.9rem; color:#d1d5db;">
                    Model decision threshold: <b>{threshold:.2f}</b><br>
                    If probability ≥ threshold, application is flagged as <b>fraud</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Right: quick interpretation ----
    with right:
        st.markdown("**Quick Interpretation**")
        if is_fraud:
            st.error(
                "The application is **likely fraudulent** based on historical patterns. "
                "Recommend manual review, document verification and transaction scrutiny."
            )
        else:
            st.success(
                "The application is **likely legitimate**. However, this is a model "
                "estimate and should be used together with other risk controls."
            )

    st.caption(
        "Note: This dashboard is for educational purposes only and should not be used as the sole "
        "basis for real-world credit or compliance decisions."
    )
