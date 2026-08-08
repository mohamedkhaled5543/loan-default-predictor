"""
Loan Risk Intelligence — Loan Default Prediction Dashboard
Built to match the exact preprocessing used in loan_decision_tree_classifier.ipynb
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import hstack

# ----------------------------------------------------------------------------
# CONFIG — must match the training notebook exactly
# ----------------------------------------------------------------------------
MODEL_PATH = "best_decision_tree.pkl"
ENCODER_PATH = "onehot_encoder.pkl"
TREE_IMAGE_PATH = "decision_tree_view.png"
DATA_PATH = "LoanDataset - LoansDatasest.csv"  # optional, only used for the loan-vs-income chart

# Order matters — this is the exact column order used to build X_train_final
NUM_COLS = [
    "customer_age", "customer_income", "employment_duration",
    "loan_grade", "loan_amnt", "loan_int_rate", "term_years", "cred_hist_length",
]
CAT_COLS = ["home_ownership", "loan_intent", "historical_default"]

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

# Update these to match your final classification_report before publishing
MODEL_METRICS = {
    "accuracy": 0.965,
    "precision": 0.96,
    "recall": 0.87,
    "f1": 0.91,
    "train_size": 25952,
    "test_size": 6489,
}

# ----------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Risk Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background-color: #0b0f19;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .app-header {
        padding: 1.4rem 1.8rem;
        background: linear-gradient(135deg, #141a2b 0%, #1c2338 100%);
        border: 1px solid #262d42;
        border-radius: 14px;
        margin-bottom: 1.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .app-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f4f6fb;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.92rem;
        color: #8b93a7;
        margin-top: 0.2rem;
    }
    .status-pill {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        background: #101627;
        border: 1px solid #263049;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-size: 0.82rem;
        color: #8ce99a;
        font-weight: 500;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #40c463;
        box-shadow: 0 0 8px #40c463;
    }
    .card {
        background: #141a2b;
        border: 1px solid #262d42;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: #141a2b;
        border: 1px solid #262d42;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        text-align: left;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #8b93a7;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f4f6fb;
    }
    .result-card-safe {
        background: linear-gradient(135deg, #0f2e1d 0%, #143824 100%);
        border: 1px solid #1f5c3a;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-card-risk {
        background: linear-gradient(135deg, #2e0f0f 0%, #3a1414 100%);
        border: 1px solid #5c1f1f;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-label-safe {
        color: #6fd694;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .result-label-risk {
        color: #f08080;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .result-status {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f4f6fb;
        margin: 0.5rem 0;
    }
    .result-prob {
        color: #b6bdcc;
        font-size: 1rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #0e1320;
        border-right: 1px solid #1e2436;
    }
    div[data-testid="stMetricValue"] {
        color: #f4f6fb;
    }
    .stButton > button {
        background: #4f6df5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton > button:hover {
        background: #3d59d8;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f4f6fb;
        margin: 1.6rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, encoder
    except Exception:
        return None, None


@st.cache_data
def load_reference_data():
    """Optional — used only for the loan-vs-income comparison chart."""
    if not os.path.exists(DATA_PATH):
        return None
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception:
        return None


# ----------------------------------------------------------------------------
# PREPROCESSING — mirrors the training notebook exactly
# ----------------------------------------------------------------------------
def prepare_input(raw: dict) -> pd.DataFrame:
    row = {
        "customer_age": raw["age"],
        "customer_income": raw["income"],
        "employment_duration": raw["employment_duration"],
        "loan_grade": GRADE_MAP[raw["loan_grade"]],
        "loan_amnt": raw["loan_amnt"],
        "loan_int_rate": raw["loan_int_rate"],
        "term_years": raw["term_years"],
        "cred_hist_length": raw["cred_hist_length"],
        "home_ownership": raw["home_ownership"],
        "loan_intent": raw["loan_intent"],
        "historical_default": raw["historical_default"],
    }
    return pd.DataFrame([row])


def predict_risk(input_df: pd.DataFrame, model, encoder):
    num_part = input_df[NUM_COLS].to_numpy()
    cat_part = encoder.transform(input_df[CAT_COLS])
    X_final = hstack([num_part, cat_part])

    prediction = model.predict(X_final)[0]
    proba = model.predict_proba(X_final)[0]
    classes = list(model.classes_)
    default_idx = classes.index("DEFAULT")
    default_proba = float(proba[default_idx])

    return prediction, default_proba, proba, classes


def calculate_risk_score(default_proba: float):
    score = round(default_proba * 100)
    if score <= 30:
        level = "Low Risk"
    elif score <= 60:
        level = "Moderate Risk"
    elif score <= 80:
        level = "High Risk"
    else:
        level = "Very High Risk"
    return score, level


def calculate_financial_metrics(income, loan_amnt, loan_int_rate, term_years):
    loan_to_income = loan_amnt / income if income > 0 else np.nan

    monthly_rate = (loan_int_rate / 100) / 12
    n_payments = max(term_years * 12, 1)

    if monthly_rate > 0:
        monthly_payment = (
            loan_amnt * (monthly_rate * (1 + monthly_rate) ** n_payments)
            / ((1 + monthly_rate) ** n_payments - 1)
        )
    else:
        monthly_payment = loan_amnt / n_payments

    return loan_to_income, monthly_payment


# ----------------------------------------------------------------------------
# DISPLAY COMPONENTS
# ----------------------------------------------------------------------------
def display_header(model_ok: bool):
    status_text = "Model Online" if model_ok else "Model Unavailable"
    dot_color = "#40c463" if model_ok else "#e05252"
    st.markdown(f"""
    <div class="app-header">
        <div>
            <p class="app-title">Loan Risk Intelligence</p>
            <p class="app-subtitle">AI-powered loan default risk assessment</p>
        </div>
        <div class="status-pill">
            <span class="status-dot" style="background:{dot_color}; box-shadow:0 0 8px {dot_color};"></span>
            {status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_prediction(prediction, default_proba, risk_score, risk_level):
    is_default = prediction == "DEFAULT"
    card_class = "result-card-risk" if is_default else "result-card-safe"
    label_class = "result-label-risk" if is_default else "result-label-safe"
    label_text = "HIGHER DEFAULT RISK" if is_default else "LOWER DEFAULT RISK"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="{label_class}">{label_text}</div>
        <div class="result-status">Predicted Status: {prediction}</div>
        <div class="result-prob">Default Probability: {default_proba*100:.1f}%  ·  Risk Score: {risk_score}/100 ({risk_level})</div>
    </div>
    """, unsafe_allow_html=True)


def display_risk_gauge(risk_score: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": " / 100", "font": {"color": "#f4f6fb"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b93a7"},
            "bar": {"color": "#4f6df5"},
            "bgcolor": "#141a2b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#1f5c3a"},
                {"range": [30, 60], "color": "#5c5320"},
                {"range": [60, 80], "color": "#5c3a1f"},
                {"range": [80, 100], "color": "#5c1f1f"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#141a2b",
        font={"color": "#f4f6fb"},
        height=260,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def display_probability_chart(default_proba: float):
    no_default_proba = 1 - default_proba
    fig = go.Figure(go.Bar(
        x=[default_proba * 100, no_default_proba * 100],
        y=["Default", "No Default"],
        orientation="h",
        marker=dict(color=["#e05252", "#40c463"]),
        text=[f"{default_proba*100:.1f}%", f"{no_default_proba*100:.1f}%"],
        textposition="auto",
    ))
    fig.update_layout(
        paper_bgcolor="#141a2b",
        plot_bgcolor="#141a2b",
        font={"color": "#f4f6fb"},
        height=220,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(range=[0, 100], showgrid=False, title="Probability (%)"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def display_kpi_cards(raw: dict):
    kpis = [
        ("Age", f"{raw['age']} yrs"),
        ("Income", f"£{raw['income']:,.0f}"),
        ("Loan Amount", f"£{raw['loan_amnt']:,.0f}"),
        ("Interest Rate", f"{raw['loan_int_rate']:.2f}%"),
        ("Loan Term", f"{raw['term_years']} yrs"),
        ("Credit History", f"{raw['cred_hist_length']} yrs"),
        ("Employment", f"{raw['employment_duration']} yrs"),
    ]
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


def display_financial_overview(raw: dict):
    loan_to_income, monthly_payment = calculate_financial_metrics(
        raw["income"], raw["loan_amnt"], raw["loan_int_rate"], raw["term_years"]
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Loan-to-Income Ratio", f"{loan_to_income:.2f}")
    with c2:
        st.metric("Est. Monthly Payment", f"£{monthly_payment:,.0f}")
    with c3:
        st.metric("Total Repayment (est.)", f"£{monthly_payment * raw['term_years'] * 12:,.0f}")


def display_feature_importance(model):
    if not hasattr(model, "feature_importances_"):
        return
    try:
        model_obj, encoder = load_artifacts()
        feature_names = list(NUM_COLS) + list(encoder.get_feature_names_out(CAT_COLS))
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(10)

        fig = px.bar(
            feat_imp.sort_values("importance"),
            x="importance", y="feature", orientation="h",
        )
        fig.update_traces(marker_color="#4f6df5")
        fig.update_layout(
            paper_bgcolor="#141a2b",
            plot_bgcolor="#141a2b",
            font={"color": "#f4f6fb"},
            height=380,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(showgrid=False, title="Importance"),
            yaxis=dict(showgrid=False, title=""),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Feature importance is unavailable for this model.")


def display_loan_vs_income(raw: dict):
    df = load_reference_data()
    if df is None or "customer_income" not in df.columns or "loan_amnt" not in df.columns:
        return

    sample = df.copy()
    sample["customer_income"] = pd.to_numeric(
        sample["customer_income"].astype(str).str.replace(",", ""), errors="coerce"
    )
    sample["loan_amnt"] = pd.to_numeric(
        sample["loan_amnt"].astype(str).str.replace("£", "").str.replace(",", ""),
        errors="coerce",
    )
    sample = sample.dropna(subset=["customer_income", "loan_amnt"])
    sample = sample[(sample["customer_income"] < sample["customer_income"].quantile(0.98))]

    fig = px.scatter(
        sample, x="customer_income", y="loan_amnt",
        opacity=0.35, color_discrete_sequence=["#4f6df5"],
    )
    fig.add_trace(go.Scatter(
        x=[raw["income"]], y=[raw["loan_amnt"]],
        mode="markers", marker=dict(color="#e05252", size=16, symbol="star"),
        name="This customer",
    ))
    fig.update_layout(
        paper_bgcolor="#141a2b",
        plot_bgcolor="#141a2b",
        font={"color": "#f4f6fb"},
        height=380,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(title="Customer Income (£)", showgrid=False),
        yaxis=dict(title="Loan Amount (£)", showgrid=False),
        showlegend=True,
    )
    st.markdown('<div class="section-title">Loan vs Income</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)


def display_model_info():
    with st.expander("About the Model"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{MODEL_METRICS['accuracy']*100:.1f}%")
        c2.metric("Precision (Default)", f"{MODEL_METRICS['precision']*100:.1f}%")
        c3.metric("Recall (Default)", f"{MODEL_METRICS['recall']*100:.1f}%")
        c4.metric("F1-score (Default)", f"{MODEL_METRICS['f1']*100:.1f}%")

        st.markdown(f"""
        **Model type:** Decision Tree Classifier (GridSearchCV-tuned)
        **Training samples:** {MODEL_METRICS['train_size']:,}
        **Test samples:** {MODEL_METRICS['test_size']:,}
        **Features used:** {len(NUM_COLS) + len(CAT_COLS)} raw inputs (numeric + one-hot encoded categoricals)
        """)


def display_tree_explorer():
    with st.expander("Explore the Decision Tree"):
        if os.path.exists(TREE_IMAGE_PATH):
            st.image(TREE_IMAGE_PATH, use_container_width=True)
        else:
            st.info("Tree visualization image not found in the app directory.")


# ----------------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------------
def main():
    model, encoder = load_artifacts()
    display_header(model_ok=model is not None)

    if model is None or encoder is None:
        st.error(
            "Model files could not be loaded. Make sure "
            f"`{MODEL_PATH}` and `{ENCODER_PATH}` are in the app directory."
        )
        st.stop()

    # ---------------- Sidebar ----------------
    st.sidebar.markdown("### Customer Information")
    age = st.sidebar.slider("Customer Age", 18, 100, 30)
    income = st.sidebar.number_input("Annual Income (£)", min_value=1000, value=50000, step=1000)
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    employment_duration = st.sidebar.slider("Employment Duration (years)", 0, 45, 5)
    historical_default_choice = st.sidebar.radio(
        "Historical Default",
        ["No previous default", "Previous default", "Not reported"],
        index=2,
    )
    historical_default_map = {
        "No previous default": "N",
        "Previous default": "Y",
        "Not reported": "Unknown",
    }
    cred_hist_length = st.sidebar.slider("Credit History Length (years)", 1, 30, 4)

    st.sidebar.markdown("### Loan Information")
    loan_amnt = st.sidebar.number_input("Loan Amount (£)", min_value=500, value=10000, step=500)
    loan_int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 11.0, 0.1)
    term_years = st.sidebar.slider("Loan Term (years)", 1, 10, 4)
    loan_intent = st.sidebar.selectbox(
        "Loan Intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    loan_grade = st.sidebar.select_slider("Loan Grade", options=["A", "B", "C", "D", "E"], value="B")

    raw_input = {
        "age": age,
        "income": income,
        "home_ownership": home_ownership,
        "employment_duration": employment_duration,
        "historical_default": historical_default_map[historical_default_choice],
        "cred_hist_length": cred_hist_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "term_years": term_years,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
    }

    # ---------------- Prediction trigger ----------------
    st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1, 1])
    with center:
        assess_clicked = st.button("Assess Loan Risk", use_container_width=True)

    if not assess_clicked:
        st.info("Fill in the customer and loan details on the left, then click **Assess Loan Risk**.")
        return

    try:
        input_df = prepare_input(raw_input)
        prediction, default_proba, proba, classes = predict_risk(input_df, model, encoder)
    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}")
        return

    risk_score, risk_level = calculate_risk_score(default_proba)

    # ---------------- Result ----------------
    display_prediction(prediction, default_proba, risk_score, risk_level)

    st.markdown('<div class="section-title">Risk Score</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        display_risk_gauge(risk_score)
    with g2:
        display_probability_chart(default_proba)

    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
    display_kpi_cards(raw_input)

    st.markdown('<div class="section-title">Financial Overview</div>', unsafe_allow_html=True)
    display_financial_overview(raw_input)

    st.markdown('<div class="section-title">Why This Prediction?</div>', unsafe_allow_html=True)
    display_feature_importance(model)

    display_loan_vs_income(raw_input)

    display_model_info()
    display_tree_explorer()


if __name__ == "__main__":
    main()
