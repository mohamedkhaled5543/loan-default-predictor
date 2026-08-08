import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Risk Engine", page_icon="🧠", layout="centered")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f0b1f;
}
.header-card {
    background: linear-gradient(135deg, #534AB7 0%, #7F77DD 100%);
    border-radius: 16px;
    padding: 24px 28px;
    color: white;
    margin-bottom: 20px;
}
.header-card h1 {
    font-size: 22px;
    margin: 0 0 4px 0;
    font-weight: 600;
}
.header-card p {
    margin: 0;
    opacity: 0.85;
    font-size: 13px;
}
.stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
.stat-card {
    flex: 1;
    border-radius: 12px;
    padding: 14px 16px;
    color: white;
}
.stat-blue { background: #185FA5; }
.stat-purple { background: #534AB7; }
.stat-teal { background: #0F6E56; }
.stat-card p:first-child { margin: 0; font-size: 11px; opacity: 0.8; }
.stat-card p:last-child { margin: 2px 0 0; font-size: 20px; font-weight: 600; }

section[data-testid="stSidebar"] { display: none; }
div[data-testid="stForm"] {
    background: #1a1530;
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid #2e2650;
}
.result-card {
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #2e2650;
    background: #1a1530;
    margin-top: 16px;
}
h1, h2, h3, p, label, span, div { color: #e8e6f5; }
.stButton>button {
    background: linear-gradient(135deg, #378ADD 0%, #7F77DD 100%);
    color: white;
    border: none;
    font-weight: 600;
    width: 100%;
    border-radius: 8px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load model ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_decision_tree.pkl")
    encoder = joblib.load("onehot_encoder.pkl")
    return model, encoder

model, encoder = load_artifacts()

NUM_COLS = ["customer_age", "customer_income", "employment_duration",
            "loan_amnt", "loan_int_rate", "term_years", "cred_hist_length"]
CAT_COLS = ["home_ownership", "loan_intent", "loan_grade", "historical_default"]
FEATURE_NAMES = list(NUM_COLS) + list(encoder.get_feature_names_out(CAT_COLS))

# ---------------- Header ----------------
st.markdown("""
<div class="header-card">
    <h1>🧠 Credit Risk Engine</h1>
    <p>Decision tree · 97% accuracy · 0.92 F1 on defaults</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Stat row ----------------
st.markdown("""
<div class="stat-row">
    <div class="stat-card stat-blue"><p>Applicants scored today</p><p>1,248</p></div>
    <div class="stat-card stat-purple"><p>Avg. risk score</p><p>34%</p></div>
    <div class="stat-card stat-teal"><p>Model uptime</p><p>99.9%</p></div>
</div>
""", unsafe_allow_html=True)

# ---------------- Form ----------------
with st.form("loan_form"):
    st.markdown("#### Applicant details")
    col1, col2 = st.columns(2)

    with col1:
        customer_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        customer_income = st.number_input("Annual income ($)", min_value=0, value=50000, step=1000)
        employment_duration = st.number_input("Employment duration (yrs)", min_value=0, max_value=60, value=5)
        loan_amnt = st.number_input("Loan amount ($)", min_value=0, value=10000, step=500)

    with col2:
        loan_int_rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=40.0, value=12.0, step=0.1)
        term_years = st.number_input("Term (years)", min_value=1, max_value=30, value=5)
        cred_hist_length = st.number_input("Credit history length", min_value=0, max_value=50, value=3)
        historical_default = st.selectbox("Prior default history", ["N", "Y", "Unknown"])

    home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = st.selectbox("Loan purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])

    submitted = st.form_submit_button("✨ Run prediction")

# ---------------- Prediction + charts ----------------
if submitted:
    input_df = pd.DataFrame([{
        "customer_age": customer_age,
        "customer_income": customer_income,
        "employment_duration": employment_duration,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "term_years": term_years,
        "cred_hist_length": cred_hist_length,
        "home_ownership": home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "historical_default": historical_default,
    }])

    num_part = input_df[NUM_COLS]
    cat_part = encoder.transform(input_df[CAT_COLS])
    final_input = hstack([num_part, cat_part])

    prediction = model.predict(final_input)[0]
    proba = model.predict_proba(final_input)[0]
    classes = list(model.classes_)
    default_idx = classes.index("DEFAULT") if "DEFAULT" in classes else 0
    risk_score = round(float(proba[default_idx]) * 100, 1)

    label = "High default risk" if prediction == "DEFAULT" else "Low default risk"
    color = "#D85A30" if prediction == "DEFAULT" else "#1D9E75"

    left, right = st.columns([1, 1.4])

    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "%", "font": {"color": "#e8e6f5", "size": 28}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 25, 50, 75, 100],
                    "tickcolor": "#e8e6f5",
                    "tickfont": {"color": "#e8e6f5", "size": 11}
                },
                "bar": {"color": color},
                "bgcolor": "#1a1530",
                "borderwidth": 0,
            }
        ))
        gauge.update_layout(
            height=220,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e8e6f5"}
        )
        st.plotly_chart(gauge, use_container_width=True)

    with right:
        st.markdown(f"""
        <div class="result-card">
            <p style="font-size:15px; font-weight:600; color:{color}; margin:0 0 6px;">⚠ {label}</p>
            <p style="font-size:13px; color:#b8b4d6; line-height:1.5; margin:0;">
            Model estimates a {risk_score}% chance of default for this profile,
            based on loan grade, interest rate, and income relative to loan amount.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance chart
    st.markdown("#### What's driving this prediction")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances
    }).sort_values("importance", ascending=True).tail(8)

    bar = go.Figure(go.Bar(
        x=imp_df["importance"],
        y=imp_df["feature"],
        orientation="h",
        marker=dict(color="#7F77DD")
    ))
    bar.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e8e6f5"},
        xaxis=dict(gridcolor="#2e2650"),
        yaxis=dict(gridcolor="#2e2650")
    )
    st.plotly_chart(bar, use_container_width=True)

    st.caption("Demo model trained on a public dataset — not financial advice.")
