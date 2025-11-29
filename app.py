import time
import json
import joblib
import pandas as pd
import streamlit as st

# =========================
#  Load model bundle & schema
# =========================
bundle = joblib.load("model.pkl")
model = bundle["model"]
threshold = bundle["threshold"]

with open("feature_columns.json", "r") as f:
    schema = json.load(f)

numeric_features = schema.get("numeric", [])
categorical_schema = schema.get("categorical", {})

# =========================
#  Page configuration
# =========================
st.set_page_config(
    page_title="Loan Fraud Detection",
    page_icon="🕵️",
    layout="wide"
)

# =========================
#  Light-blue professional theme (CSS)
# =========================
st.markdown(
    """
    <style>
    /* Main background */
    [data-testid="stAppViewContainer"] {
        background: #f4f7fb;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #e6f0ff;
    }
    /* Headers */
    h1, h2, h3 {
        color: #0f4c81;
    }
    /* Section titles inside markdown */
    h1 span, h2 span, h3 span {
        color: #0f4c81;
    }
    /* Buttons */
    div.stButton > button {
        background-color: #0f4c81;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.4rem 0.75rem;
    }
    div.stButton > button:hover {
        background-color: #0c3a63;
        color: white;
    }
    /* Metric label */
    [data-testid="stMetricLabel"] > div {
        color: #0f4c81;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
#  Sidebar
# =========================
st.sidebar.title("🕵️ Loan Fraud Detection")

st.sidebar.markdown(
    """
    This dashboard estimates the **likelihood of fraud** for a loan application  
    using a **Random Forest model** trained on historical applications  
    and transaction behaviour.
    
    **How to use:**
    1. Fill in the key fields on the right  
    2. Click **Predict Fraud**  
    3. Review the probability and risk band
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Legend**")
st.sidebar.markdown("🟢 **Low** : p < 20%")
st.sidebar.markdown("🟠 **Medium** : 20% ≤ p < 50%")
st.sidebar.markdown("🔴 **High** : p ≥ 50%")

st.sidebar.markdown("---")
st.sidebar.caption("Academic project – not for real-world credit decisions.")

# =========================
#  Header
# =========================
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0.2rem;">
        Loan Fraud Risk Dashboard
    </h1>
    <p style="text-align:center; color:gray; font-size:0.95rem;">
        Uses applicant profile, loan details and key transactional indicators to estimate
        the probability that an application is fraudulent.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================
#  Input Form (only 12 top features)
# =========================
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # -------- Left column: Applicant & Loan --------
    with col1:
        st.markdown("### 👤 Applicant & Loan Details")

        applicant_age = st.number_input(
            "Applicant Age (years)", min_value=18, max_value=99, value=30
        )
        monthly_income = st.number_input(
            "Monthly Income (₹)", min_value=0, value=50_000
        )
        loan_tenure_months = st.number_input(
            "Loan Tenure (months)", min_value=1, value=60
        )
        interest_rate_offered = st.number_input(
            "Interest Rate (%)", min_value=0.0, value=12.0
        )
        number_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=0
        )
        existing_emis_monthly = st.number_input(
            "Existing EMIs (monthly, ₹)", min_value=0, value=0
        )

    # -------- Right column: Risk & Transactions --------
    with col2:
        st.markdown("### 📊 Risk & Transaction Indicators")

        debt_to_income_ratio = st.number_input(
            "Debt-to-Income Ratio", min_value=0.0, value=0.30, format="%.2f"
        )
        txn_failed_count = st.number_input(
            "Failed Transactions (count)", min_value=0, value=0
        )
        txn_avg_amount = st.number_input(
            "Average Transaction Amount (₹)", min_value=0, value=2_000
        )
        txn_total_amount = st.number_input(
            "Total Transaction Amount (₹)", min_value=0, value=40_000
        )
        txn_avg_balance_after = st.number_input(
            "Average Balance After Transaction (₹)", min_value=0, value=15_000
        )
        cibil_score = st.number_input(
            "CIBIL Score", min_value=300, max_value=900, value=750
        )

    st.markdown("")
    submit_col, _, _ = st.columns([1, 3, 3])
    with submit_col:
        submitted = st.form_submit_button("🔍 Predict Fraud", use_container_width=True)

# =========================
#  Prediction Logic
# =========================
if submitted:

    # ---- Progress & spinner (fake 2-second run to feel realistic) ----
    with st.spinner("Running fraud risk model..."):
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        for i in range(0, 101, 5):
            time.sleep(0.1)   # 0.1 * 20 ≈ 2 seconds total
            progress_bar.progress(i)

        # ---- 1) Start with default values for ALL model features ----
        input_data: dict[str, object] = {}

        # numeric: default 0.0
        for col in numeric_features:
            input_data[col] = 0.0

        # categorical: default to first known category
        for col, options in categorical_schema.items():
            input_data[col] = options[0] if options else None

        # ---- 2) Overwrite with values from form (12 top features) ----
        input_data.update({
            "txn_failed_count": txn_failed_count,
            "applicant_age": applicant_age,
            "monthly_income": monthly_income,
            "loan_tenure_months": loan_tenure_months,
            "debt_to_income_ratio": debt_to_income_ratio,
            "txn_avg_amount": txn_avg_amount,
            "txn_total_amount": txn_total_amount,
            "txn_avg_balance_after": txn_avg_balance_after,
            "existing_emis_monthly": existing_emis_monthly,
            "number_of_dependents": number_of_dependents,
            "interest_rate_offered": interest_rate_offered,
            "cibil_score": cibil_score,
        })

        # ---- 3) Build DataFrame & predict ----
        X_input = pd.DataFrame([input_data])
        prob_fraud = float(model.predict_proba(X_input)[0][1])
        is_fraud = prob_fraud >= threshold

    # clear progress bar after done
    progress_placeholder.empty()

    # =========================
    #  Result Display
    # =========================
    st.markdown("---")
    st.subheader("Prediction Summary")

    left, right = st.columns([1.2, 1])

    # ---- Left: Probability + Risk badge ----
    with left:
        st.metric("Fraud Probability", f"{prob_fraud * 100:.2f}%")

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
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                ">
                <span style="font-size:1.1rem;">{emoji} <b>{risk_label}</b></span><br>
                <span style="font-size:0.9rem; color:#4b5563;">
                    Model decision threshold: <b>{threshold:.2f}</b><br>
                    If probability ≥ threshold, the application is flagged as <b>fraud</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Right: Interpretation text ----
    with right:
        st.markdown("**Interpretation**")
        if is_fraud:
            st.error(
                "The model flags this application as **likely fraudulent**. "
                "Recommend detailed manual review, document verification, "
                "and closer inspection of recent transactions."
            )
        else:
            st.success(
                "The model considers this application **likely legitimate**. "
                "It is still recommended to combine this with standard KYC "
                "and risk checks."
            )

    st.caption(
        "Note: This dashboard is designed for learning and demonstration only. "
        "It should not be used as the sole basis for real-world lending decisions."
    )
