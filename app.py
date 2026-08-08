"""
Loan Risk Intelligence — Loan Default Prediction Dashboard
Matches the exact preprocessing pipeline used in loan_decision_tree_classifier.ipynb
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
MODEL_PATH = "best_decision_tree_calibrated.pkl"   # used for predictions/probabilities
TREE_MODEL_PATH = "best_decision_tree.pkl"          # original tree, used for importance + plot
ENCODER_PATH = "onehot_encoder.pkl"
TREE_IMAGE_PATH = "decision_tree_view.png"
DATA_PATH = "LoanDataset - LoansDatasest.csv"       # optional, only used for the loan-vs-income chart

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
# PAGE CONFIG + STYLE — clean light fintech theme
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background-color: #f6f8fa; }
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    .app-header {
        padding: 1.5rem 2rem;
        background: linear-gradient(120deg, #0f9d78 0%, #0b7d8c 100%);
        border-radius: 16px;
        margin-bottom: 1.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px rgba(15, 157, 120, 0.18);
    }
    .app-title { font-size: 1.8rem; font-weight: 800; color: #ffffff; margin: 0; letter-spacing: -0.02em; }
    .app-subtitle { font-size: 0.95rem; color: #e3fbf3; margin-top: 0.2rem; }
    .status-pill {
        display: flex; align-items: center; gap: 0.45rem;
        background: rgba(255,255,255,0.15);
        padding: 0.45rem 1rem; border-radius: 999px;
        font-size: 0.82rem; color: #ffffff; font-weight: 600;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #baffc9; box-shadow: 0 0 8px #baffc9; }

    .card {
        background: #ffffff; border: 1px solid #e6ebf0; border-radius: 16px;
        padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(20, 30, 40, 0.04);
    }
    .kpi-card {
        background: #ffffff; border: 1px solid #e6ebf0; border-radius: 14px;
        padding: 1rem 1.1rem; text-align: left;
        box-shadow: 0 2px 8px rgba(20, 30, 40, 0.03);
    }
    .kpi-label { font-size: 0.72rem; color: #7a8794; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; }
    .kpi-value { font-size: 1.3rem; font-weight: 800; color: #16232e; }

    .result-card-safe {
        background: #eafbf3; border: 1px solid #a9e8cb; border-radius: 18px;
        padding: 2.2rem; text-align: center;
        box-shadow: 0 4px 18px rgba(15, 157, 120, 0.10);
    }
    .result-card-risk {
        background: #fdecec; border: 1px solid #f3b8b8; border-radius: 18px;
        padding: 2.2rem; text-align: center;
        box-shadow: 0 4px 18px rgba(214, 60, 60, 0.10);
    }
    .result-label-safe { color: #0f9d78; font-size: 0.95rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
    .result-label-risk { color: #d63c3c; font-size: 0.95rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
    .result-status { font-size: 2.2rem; font-weight: 800; color: #16232e; margin: 0.5rem 0; }
    .result-prob { color: #4a5a68; font-size: 1.02rem; }

    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e6ebf0; }
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #16232e; }

    div[data-testid="stMetricValue"] { color: #16232e; }

    .stButton > button {
        background: #0f9d78; color: white; border: none; border-radius: 12px;
        padding: 0.8rem 1.6rem; font-weight: 700; font-size: 1.02rem;
        box-shadow: 0 4px 14px rgba(15, 157, 120, 0.25);
    }
    .stButton > button:hover { background: #0c8064; }

    .section-title { font-size: 1.2rem; font-weight: 800; color: #16232e; margin: 1.8rem 0 0.9rem 0; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return None, None, None
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        tree_model = joblib.load(TREE_MODEL_PATH) if os.path.exists(TREE_MODEL_PATH) else None
        return model, encoder, tree_model
    except Exception:
        return None, None, None


@st.cache_data
def load_reference_data():
    if not os.path.exists(DATA_PATH):
        return None
    try:
        return pd.read_csv(DATA_PATH)
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
    dot_color = "#baffc9" if model_ok else "#ffb3b3"
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
    if risk_score <= 30:
        bar_color = "#0f9d78"
    elif risk_score <= 60:
        bar_color = "#e0a530"
    elif risk_score <= 80:
        bar_color = "#e0722f"
    else:
        bar_color = "#d63c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": " / 100", "font": {"color": "#16232e", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#7a8794"},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "#ffffff",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#dff5ea"},
                {"range": [30, 60], "color": "#fbf1d9"},
                {"range": [60, 80], "color": "#fbe4d6"},
                {"range": [80, 100], "color": "#fbdada"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        font={"color": "#16232e"},
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
        marker=dict(color=["#d63c3c", "#0f9d78"]),
        text=[f"{default_proba*100:.1f}%", f"{no_default_proba*100:.1f}%"],
        textposition="auto",
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#16232e"},
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


def display_feature_importance(tree_model, encoder):
    if tree_model is None or not hasattr(tree_model, "feature_importances_"):
        st.info("Feature importance is unavailable — original tree model not found.")
        return
    try:
        feature_names = list(NUM_COLS) + list(encoder.get_feature_names_out(CAT_COLS))
        importances = tree_model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(10)

        fig = px.bar(
            feat_imp.sort_values("importance"),
            x="importance", y="feature", orientation="h",
        )
        fig.update_traces(marker_color="#0f9d78")
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"color": "#16232e"},
            height=380,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(showgrid=False, title="Importance"),
            yaxis=dict(showgrid=False, title=""),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Feature importance could not be displayed.")


def display_loan_vs_income(raw: dict):
    df = load_reference_data()
    if df is None or "customer_income" not in df.columns or "loan_amnt" not in df.columns:
        return
    try:
        sample = df.copy()
        sample["customer_income"] = pd.to_numeric(
            sample["customer_income"].astype(str).str.replace(",", ""), errors="coerce"
        )
        sample["loan_amnt"] = pd.to_numeric(
            sample["loan_amnt"].astype(str).str.replace("£", "").str.replace(",", ""),
            errors="coerce",
        )
        sample = sample.dropna(subset=["customer_income", "loan_amnt"])
        sample = sample[sample["customer_income"] < sample["customer_income"].quantile(0.98)]

        fig = px.scatter(
            sample, x="customer_income", y="loan_amnt",
            opacity=0.3, color_discrete_sequence=["#0b7d8c"],
        )
        fig.add_trace(go.Scatter(
            x=[raw["income"]], y=[raw["loan_amnt"]],
            mode="markers", marker=dict(color="#d63c3c", size=16, symbol="star"),
            name="This customer",
        ))
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"color": "#16232e"},
            height=380,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(title="Customer Income (£)", showgrid=False),
            yaxis=dict(title="Loan Amount (£)", showgrid=False),
            showlegend=True,
        )
        st.markdown('<div class="section-title">Loan vs Income</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def display_model_info():
    with st.expander("About the Model"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{MODEL_METRICS['accuracy']*100:.1f}%")
        c2.metric("Precision (Default)", f"{MODEL_METRICS['precision']*100:.1f}%")
        c3.metric("Recall (Default)", f"{MODEL_METRICS['recall']*100:.1f}%")
        c4.metric("F1-score (Default)", f"{MODEL_METRICS['f1']*100:.1f}%")
        st.markdown(f"""
        **Model type:** Decision Tree Classifier (GridSearchCV-tuned, probability-calibrated)
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
    model, encoder, tree_model = load_artifacts()
    display_header(model_ok=model is not None)

    if model is None or encoder is None:
        st.error(
            "Model files could not be loaded. Make sure "
            f"`{MODEL_PATH}` and `{ENCODER_PATH}` are in the app directory."
        )
        st.stop()

    # ---------------- Sidebar — all typeable number inputs ----------------
    st.sidebar.markdown("### Customer Information")
    age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=30, step=1)
    income = st.sidebar.number_input("Annual Income (£)", min_value=1000, value=50000, step=1000)
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    employment_duration = st.sidebar.number_input("Employment Duration (years)", min_value=0, max_value=45, value=5, step=1)
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
    cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=1, max_value=30, value=4, step=1)

    st.sidebar.markdown("### Loan Information")
    loan_amnt = st.sidebar.number_input("Loan Amount (£)", min_value=500, value=10000, step=500)
    loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.0, step=0.1, format="%.2f")
    term_years = st.sidebar.number_input("Loan Term (years)", min_value=1, max_value=10, value=4, step=1)
    loan_intent = st.sidebar.selectbox(
        "Loan Intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E"], index=1)

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
        st.info("Enter the customer and loan details on the left, then click **Assess Loan Risk**.")
        return

    try:
        input_df = prepare_input(raw_input)
        prediction, default_proba, proba, classes = predict_risk(input_df, model, encoder)
    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}")
        return

    risk_score, risk_level = calculate_risk_score(default_proba)

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
    display_feature_importance(tree_model, encoder)

    display_loan_vs_income(raw_input)

    display_model_info()
    display_tree_explorer()


if __name__ == "__main__":
    main()
