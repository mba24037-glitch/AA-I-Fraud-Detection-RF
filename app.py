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
#  DARK BLUE Theme (Fintech-style)
# =========================
st.markdown(
    """
    <style>
    
    /* Background */
    [data-testid="stAppViewContainer"] {
        background: #0a1a2f !important;
        color: #eaf4ff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d233d !important;
    }

    /* Input fields */
    .stNumberInput input, .stTextInput input {
        background-color: #1a2e49 !important;
        color: #eaf4ff !important;
        border-radius: 8px;
        border: 1px solid #3e5570;
    }

    /* Dropdowns */
    .stSelectbox div[data-baseweb="select"] * {
        background-color: #1a2e49 !important;
        color: white !important;
    }

    /* Titles & subtitles */
    h1, h2, h3, label, p, span, div {
        color: #eaf4ff !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #0077ff !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover {
        background-color: #005fcc !important;
    }

    /* Cards */
    .card {
        background-color: #11243d !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1f3a5f;
    }

    /* Metrics text */
    [data-testid="stMetricLabel"], 
    [data-testid="stMetricValue"] {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
#  Sidebar
# =========================
st.sidebar.title(" Loan Fraud Detection")

st.sidebar.markdown(
    """
    A professional dashboard estimating fraud likelihood
    using **Random Forest** and transaction indicators.

    **Steps**
    
    1️⃣ Fill in the key fields  
    2️⃣ Click **Predict Fraud**  
    3️⃣ Review fraud risk band  
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Levels**")
st.sidebar.markdown("🟢 Low Risk <20%")
st.sidebar.markdown("🟠 Medium Risk 20–50%")
st.sidebar.markdown("🔴 High Risk ≥50%")
st.sidebar.markdown("---")

# =========================
#  Header
# =========================
st.markdown(
    "<h1 style='text-align:center;'>Loan Fraud Risk Detection Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================
#  Input Form
# =========================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Applicant & Loan Details")
        applicant_age = st.number_input("Age", 18, 99, 30)
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=40000)
        loan_tenure_months = st.number_input("Loan Tenure (months)", 1, 360, 60)
        interest_rate_offered = st.number_input("Interest Rate (%)", 0.0, 30.0, 12.0)
        number_of_dependents = st.number_input("Dependents", 0, 10, 1)
        existing_emis_monthly = st.number_input("Existing EMIs (₹)", 0, 200000, 0)

    with col2:
        st.markdown("###  Transaction Indicators")
        debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 5.0, 0.35, step=0.01)
        txn_failed_count = st.number_input("Failed Transactions", 0, 50, 2)
        txn_avg_amount = st.number_input("Average Txn Amount (₹)", 0, 200000, 2000)
        txn_total_amount = st.number_input("Total Txn Amount (₹)", 0, 2000000, 50000)
        txn_avg_balance_after = st.number_input("Avg Balance After Txn (₹)", 0, 500000, 15000)
        cibil_score = st.number_input("CIBIL Score", 300, 900, 750)

    submitted = st.form_submit_button("🔍 Predict Fraud")

# =========================
#  Prediction Logic + Progress Bar
# =========================
if submitted:
    with st.spinner("Evaluating risk score..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)  # ~2 sec
            progress.progress(i + 1)

    # Create complete feature vector
    row = {col: 0.0 for col in numeric_features}
    for col, opts in categorical_schema.items():
        row[col] = opts[0] if opts else None

    row.update({
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

    X_input = pd.DataFrame([row])
    probability = float(model.predict_proba(X_input)[0][1])
    flagged = probability >= threshold

    st.markdown("---")
    st.subheader("📈 Prediction Summary")

    col_a, col_b = st.columns(2)

    # Probability + Risk Badge
    with col_a:
        st.metric("Fraud Probability", f"{probability*100:.2f}%")

        if probability < 0.20:
            st.success("🟢 **Low Risk**")
        elif probability < 0.50:
            st.warning("🟠 **Medium Risk**")
        else:
            st.error("🔴 **High Risk**")

    # Risk Statement
    with col_b:
        if flagged:
            st.error(
                "⚠️ System recommends **fraud investigation** — "
                "verify employment, financial statements & device history."
            )
        else:
            st.success(
                "✔ Likely **legitimate** — still cross-verify with KYC & credit rules."
            )

    st.caption("⚠ AI model usage for academic & demonstration purposes only.")
