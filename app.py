"""
Loan Risk Intelligence — Neon Edition
Same preprocessing pipeline as loan_decision_tree_classifier.ipynb.
Only the UI/UX layer has been rebuilt: dark neon theme + richer Plotly visuals.
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
MODEL_PATH = "best_decision_tree_calibrated.pkl"
TREE_MODEL_PATH = "best_decision_tree.pkl"
ENCODER_PATH = "onehot_encoder.pkl"
TREE_IMAGE_PATH = "decision_tree_view.png"
DATA_PATH = "LoanDataset - LoansDatasest.csv"

NUM_COLS = [
    "customer_age", "customer_income", "employment_duration",
    "loan_grade", "loan_amnt", "loan_int_rate", "term_years", "cred_hist_length",
]
CAT_COLS = ["home_ownership", "loan_intent", "historical_default"]
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

MODEL_METRICS = {
    "accuracy": 0.965, "precision": 0.96, "recall": 0.87, "f1": 0.91,
    "train_size": 25952, "test_size": 6489,
}

# Neon palette
NEON_CYAN = "#00f5d4"
NEON_PINK = "#ff2d95"
NEON_PURPLE = "#9b5de5"
NEON_YELLOW = "#f9f871"
NEON_ORANGE = "#ff9f1c"
NEON_RED = "#ff3860"
BG_DARK = "#0b0e1a"
BG_CARD = "#131829"
GRID = "rgba(255,255,255,0.06)"
TEXT_MAIN = "#eaf2ff"
TEXT_DIM = "#8a93a8"

# ----------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Loan Risk Intelligence — Neon", layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{
        background:
            radial-gradient(circle at 15% 0%, rgba(0,245,212,0.06), transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(255,45,149,0.06), transparent 40%),
            {BG_DARK};
    }}
    html, body, [class*="css"] {{ font-family: 'Inter','Segoe UI',sans-serif; color: {TEXT_MAIN}; }}

    .app-header {{
        padding: 1.5rem 2rem;
        background: linear-gradient(120deg, rgba(0,245,212,0.09), rgba(155,93,229,0.09));
        border: 1px solid rgba(0,245,212,0.22);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        display: flex; justify-content: space-between; align-items: center;
        box-shadow: 0 0 18px rgba(0,245,212,0.07);
    }}
    .app-title {{
        font-size: 1.9rem; font-weight: 900; margin: 0; letter-spacing: -0.02em;
        background: linear-gradient(90deg, {NEON_CYAN}, {NEON_PINK});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .app-subtitle {{ font-size: 0.92rem; color: {TEXT_DIM}; margin-top: 0.25rem; }}
    .status-pill {{
        display: flex; align-items: center; gap: 0.5rem;
        background: rgba(255,255,255,0.03); padding: 0.5rem 1.1rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(0,245,212,0.28); color: {NEON_CYAN};
    }}
    .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {NEON_CYAN}; box-shadow: 0 0 6px {NEON_CYAN}; }}

    .card {{
        background: {BG_CARD}; border: 1px solid rgba(255,255,255,0.07); border-radius: 16px;
        padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }}
    .card-title {{ font-size: 1.05rem; font-weight: 800; color: {TEXT_MAIN}; margin-bottom: 0.9rem; }}

    .kpi-card {{
        background: {BG_CARD}; border: 1px solid rgba(0,245,212,0.12); border-radius: 14px;
        padding: 1rem 1.1rem; text-align: left; transition: 0.2s;
    }}
    .kpi-card:hover {{ border-color: rgba(0,245,212,0.35); box-shadow: 0 0 10px rgba(0,245,212,0.10); }}
    .kpi-label {{ font-size: 0.7rem; color: {TEXT_DIM}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }}
    .kpi-value {{ font-size: 1.25rem; font-weight: 800; color: {TEXT_MAIN}; }}

    .result-card-safe {{
        background: linear-gradient(135deg, rgba(0,245,212,0.09), rgba(0,245,212,0.02));
        border: 1px solid rgba(0,245,212,0.75); border-radius: 20px; padding: 2.1rem; text-align: center;
        box-shadow: 0 0 26px rgba(0,245,212,0.13);
    }}
    .result-card-risk {{
        background: linear-gradient(135deg, rgba(255,56,96,0.11), rgba(255,56,96,0.02));
        border: 1px solid rgba(255,56,96,0.75); border-radius: 20px; padding: 2.1rem; text-align: center;
        box-shadow: 0 0 26px rgba(255,56,96,0.13);
    }}
    .result-label-safe {{ color: {NEON_CYAN}; font-size: 0.95rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
    .result-label-risk {{ color: {NEON_RED}; font-size: 0.95rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
    .result-status {{ font-size: 2.3rem; font-weight: 900; color: {TEXT_MAIN}; margin: 0.5rem 0; }}
    .result-prob {{ color: {TEXT_DIM}; font-size: 1.0rem; }}

    section[data-testid="stSidebar"] {{ background-color: #0d1120; border-right: 1px solid rgba(0,245,212,0.10); }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{ color: {NEON_CYAN}; }}
    div[data-testid="stMetricValue"] {{ color: {TEXT_MAIN}; }}
    div[data-testid="stMetricLabel"] {{ color: {TEXT_DIM}; }}

    .stButton > button {{
        background: linear-gradient(90deg, {NEON_CYAN}, {NEON_PURPLE}); color: #06121c; border: none;
        border-radius: 12px; padding: 0.85rem 1.6rem; font-weight: 800; font-size: 1.05rem;
        box-shadow: 0 0 14px rgba(0,245,212,0.25);
    }}
    .stButton > button:hover {{ box-shadow: 0 0 20px rgba(0,245,212,0.4); transform: translateY(-1px); }}

    .section-title {{
        font-size: 1.15rem; font-weight: 800; margin: 1.8rem 0 0.9rem 0; color: {TEXT_MAIN};
        border-left: 3px solid {NEON_CYAN}; padding-left: 0.7rem;
    }}
    hr {{ border-color: rgba(255,255,255,0.07); }}

    /* ---- Applicant-details input card (matches reference: label above, stepper input) ---- */
    .input-card {{
        background: {BG_CARD}; border: 1px solid rgba(255,255,255,0.07); border-radius: 18px;
        padding: 1.5rem 1.7rem 0.6rem 1.7rem; margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }}
    .input-card-title {{
        font-size: 1.25rem; font-weight: 800; color: {TEXT_MAIN}; margin-bottom: 1.1rem;
    }}
    .input-card div[data-testid="stNumberInput"] label p {{
        font-size: 0.82rem; color: {TEXT_DIM}; font-weight: 500;
    }}
    .input-card div[data-testid="stNumberInput"] > div > div {{
        background-color: #171c30; border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
    }}
    .input-card div[data-testid="stNumberInput"] input {{
        color: {TEXT_MAIN}; font-weight: 600; font-size: 0.95rem;
    }}
    .input-card div[data-testid="stNumberInput"] button {{
        background-color: #1d2338; border-color: rgba(255,255,255,0.08); color: {TEXT_DIM};
    }}
    .input-card div[data-testid="stNumberInput"] button:hover {{
        background-color: #232a44; color: {NEON_CYAN};
    }}
    div[data-testid="stSelectbox"] label p, div[data-testid="stRadio"] label p {{
        font-size: 0.82rem; color: {TEXT_DIM}; font-weight: 500;
    }}
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
        level, color = "Low Risk", NEON_CYAN
    elif score <= 60:
        level, color = "Moderate Risk", NEON_YELLOW
    elif score <= 80:
        level, color = "High Risk", NEON_ORANGE
    else:
        level, color = "Very High Risk", NEON_RED
    return score, level, color


def amortization_schedule(loan_amnt, loan_int_rate, term_years):
    monthly_rate = (loan_int_rate / 100) / 12
    n = max(term_years * 12, 1)
    if monthly_rate > 0:
        payment = loan_amnt * (monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)
    else:
        payment = loan_amnt / n
    balance = loan_amnt
    balances, principal_paid, interest_paid = [], [], []
    cum_principal = 0
    for _ in range(int(n)):
        interest = balance * monthly_rate
        principal = payment - interest
        balance = max(balance - principal, 0)
        cum_principal += principal
        balances.append(balance)
        principal_paid.append(cum_principal)
        interest_paid.append(payment * len(balances) - cum_principal)
    return payment, balances, principal_paid, interest_paid


# ----------------------------------------------------------------------------
# DISPLAY COMPONENTS
# ----------------------------------------------------------------------------
def display_header(model_ok: bool):
    status_text = "Model Online" if model_ok else "Model Unavailable"
    dot_color = NEON_CYAN if model_ok else NEON_RED
    st.markdown(f"""
    <div class="app-header">
        <div>
            <p class="app-title">⚡ Loan Risk Intelligence</p>
            <p class="app-subtitle">AI-powered loan default risk assessment — neon edition</p>
        </div>
        <div class="status-pill">
            <span class="status-dot" style="background:{dot_color}; box-shadow:0 0 10px {dot_color};"></span>
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
        <div class="result-prob">Default Probability: {default_proba*100:.1f}% &nbsp;·&nbsp; Risk Score: {risk_score}/100 ({risk_level})</div>
    </div>
    """, unsafe_allow_html=True)

    if default_proba <= 0.0 or default_proba >= 1.0:
        st.warning(
            "This probability came out as a hard 0% or 100% — a sign the loaded model isn't "
            "calibrated (pure decision-tree leaves do this). Retrain with `CalibratedClassifierCV` "
            f"and overwrite `{MODEL_PATH}` to get smooth in-between percentages."
        )


def neon_layout(fig, height=300):
    fig.update_layout(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font={"color": TEXT_MAIN, "family": "Inter"},
        height=height, margin=dict(t=30, b=20, l=20, r=20),
    )
    return fig


def display_risk_gauge(risk_score: int, bar_color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": " / 100", "font": {"color": TEXT_MAIN, "size": 42}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_DIM},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": BG_CARD,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(0,245,212,0.12)"},
                {"range": [30, 60], "color": "rgba(249,248,113,0.10)"},
                {"range": [60, 80], "color": "rgba(255,159,28,0.12)"},
                {"range": [80, 100], "color": "rgba(255,56,96,0.14)"},
            ],
        },
    ))
    st.plotly_chart(neon_layout(fig, 270), use_container_width=True)


def display_probability_chart(default_proba: float):
    no_default_proba = 1 - default_proba
    fig = go.Figure(go.Bar(
        x=[default_proba * 100, no_default_proba * 100],
        y=["Default", "No Default"],
        orientation="h",
        marker=dict(color=[NEON_RED, NEON_CYAN], line=dict(width=0)),
        text=[f"{default_proba*100:.1f}%", f"{no_default_proba*100:.1f}%"],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor=GRID, title="Probability (%)"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(neon_layout(fig, 270), use_container_width=True)


def display_kpi_cards(raw: dict):
    kpis = [
        ("Age", f"{raw['age']} yrs"), ("Income", f"£{raw['income']:,.0f}"),
        ("Loan Amount", f"£{raw['loan_amnt']:,.0f}"), ("Interest Rate", f"{raw['loan_int_rate']:.2f}%"),
        ("Loan Term", f"{raw['term_years']} yrs"), ("Credit History", f"{raw['cred_hist_length']} yrs"),
        ("Employment", f"{raw['employment_duration']} yrs"),
    ]
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)


def display_amortization_chart(raw: dict):
    payment, balances, principal_paid, interest_paid = amortization_schedule(
        raw["loan_amnt"], raw["loan_int_rate"], raw["term_years"]
    )
    months = list(range(1, len(balances) + 1))

    c1, c2, c3 = st.columns(3)
    c1.metric("Est. Monthly Payment", f"£{payment:,.0f}")
    c2.metric("Loan-to-Income Ratio", f"{(raw['loan_amnt']/raw['income']) if raw['income'] else 0:.2f}")
    c3.metric("Total Repayment (est.)", f"£{payment * len(months):,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=balances, mode="lines", name="Remaining Balance",
        line=dict(color=NEON_CYAN, width=3), fill="tozeroy", fillcolor="rgba(0,245,212,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=principal_paid, mode="lines", name="Principal Paid",
        line=dict(color=NEON_PURPLE, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=months, y=interest_paid, mode="lines", name="Interest Paid",
        line=dict(color=NEON_PINK, width=3, dash="dot"),
    ))
    fig.update_layout(
        xaxis=dict(title="Month", showgrid=True, gridcolor=GRID),
        yaxis=dict(title="£", showgrid=True, gridcolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(neon_layout(fig, 400), use_container_width=True)


def display_feature_importance(tree_model, encoder):
    if tree_model is None or not hasattr(tree_model, "feature_importances_"):
        st.info("Feature importance is unavailable — original tree model not found.")
        return
    try:
        feature_names = list(NUM_COLS) + list(encoder.get_feature_names_out(CAT_COLS))
        importances = tree_model.feature_importances_
        feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}) \
            .sort_values("importance", ascending=False).head(10)

        fig = px.bar(feat_imp.sort_values("importance"), x="importance", y="feature", orientation="h")
        fig.update_traces(marker_color=NEON_CYAN, marker_line_width=0)
        fig.update_layout(
            xaxis=dict(showgrid=True, gridcolor=GRID, title="Importance"),
            yaxis=dict(showgrid=False, title=""),
        )
        st.plotly_chart(neon_layout(fig, 400), use_container_width=True)
    except Exception:
        st.info("Feature importance could not be displayed.")


def display_profile_radar(raw: dict, df: pd.DataFrame):
    """Radar comparing this applicant's normalized profile vs the dataset average."""
    if df is None:
        return
    try:
        cols_map = {
            "customer_age": raw["age"], "customer_income": raw["income"],
            "loan_amnt": raw["loan_amnt"], "loan_int_rate": raw["loan_int_rate"],
            "cred_hist_length": raw["cred_hist_length"], "employment_duration": raw["employment_duration"],
        }
        labels, applicant_vals, avg_vals = [], [], []
        for col, val in cols_map.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("£", ""), errors="coerce").dropna()
            if series.empty or series.max() == series.min():
                continue
            norm_applicant = (val - series.min()) / (series.max() - series.min())
            norm_avg = (series.mean() - series.min()) / (series.max() - series.min())
            labels.append(col.replace("_", " ").title())
            applicant_vals.append(max(0, min(1, norm_applicant)) * 100)
            avg_vals.append(max(0, min(1, norm_avg)) * 100)

        if not labels:
            return

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=applicant_vals + [applicant_vals[0]], theta=labels + [labels[0]],
            fill="toself", name="This Applicant",
            line=dict(color=NEON_CYAN, width=2), fillcolor="rgba(0,245,212,0.18)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=avg_vals + [avg_vals[0]], theta=labels + [labels[0]],
            fill="toself", name="Dataset Average",
            line=dict(color=NEON_PINK, width=2), fillcolor="rgba(255,45,149,0.10)",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor=BG_CARD,
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID, color=TEXT_DIM),
                angularaxis=dict(gridcolor=GRID, color=TEXT_MAIN),
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        )
        st.markdown('<div class="section-title">Applicant Profile vs Average</div>', unsafe_allow_html=True)
        st.plotly_chart(neon_layout(fig, 420), use_container_width=True)
    except Exception:
        pass


def display_loan_vs_income(raw: dict, df: pd.DataFrame):
    if df is None or "customer_income" not in df.columns or "loan_amnt" not in df.columns:
        return
    try:
        sample = df.copy()
        sample["customer_income"] = pd.to_numeric(sample["customer_income"].astype(str).str.replace(",", ""), errors="coerce")
        sample["loan_amnt"] = pd.to_numeric(sample["loan_amnt"].astype(str).str.replace("£", "").str.replace(",", ""), errors="coerce")
        sample = sample.dropna(subset=["customer_income", "loan_amnt"])
        sample = sample[sample["customer_income"] < sample["customer_income"].quantile(0.98)]

        fig = px.scatter(sample, x="customer_income", y="loan_amnt", opacity=0.35,
                          color_discrete_sequence=[NEON_PURPLE])
        fig.add_trace(go.Scatter(
            x=[raw["income"]], y=[raw["loan_amnt"]], mode="markers",
            marker=dict(color=NEON_RED, size=18, symbol="star", line=dict(color=NEON_YELLOW, width=1)),
            name="This customer",
        ))
        fig.update_layout(
            xaxis=dict(title="Customer Income (£)", showgrid=True, gridcolor=GRID),
            yaxis=dict(title="Loan Amount (£)", showgrid=True, gridcolor=GRID),
            showlegend=True,
        )
        st.markdown('<div class="section-title">Loan vs Income (Portfolio Context)</div>', unsafe_allow_html=True)
        st.plotly_chart(neon_layout(fig, 400), use_container_width=True)
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
        st.error(f"Model files could not be loaded. Make sure `{MODEL_PATH}` and `{ENCODER_PATH}` are in the app directory.")
        st.stop()

    # ---------------- Main-page "Applicant details" card — number inputs with -/+ steppers ----------------
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Applicant details</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    with r1c2:
        loan_int_rate = st.number_input("Interest rate (%)", min_value=5.0, max_value=25.0, value=12.0, step=0.5, format="%.2f")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        income = st.number_input("Annual income ($)", min_value=1000, max_value=1000000, value=50000, step=1000)
    with r2c2:
        term_years = st.number_input("Term (years)", min_value=1, max_value=10, value=5, step=1)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        employment_duration = st.number_input("Employment duration (yrs)", min_value=0, max_value=45, value=5, step=1)
    with r3c2:
        cred_hist_length = st.number_input("Credit history length", min_value=1, max_value=30, value=3, step=1)

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        loan_amnt = st.number_input("Loan amount ($)", min_value=500, max_value=500000, value=10000, step=500)
    with r4c2:
        loan_grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E"], index=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Second card — categorical details ----------------
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Loan &amp; history</div>', unsafe_allow_html=True)
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    with r5c2:
        loan_intent = st.selectbox(
            "Loan intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
    historical_default_choice = st.radio(
        "Historical default", ["No previous default", "Previous default", "Not reported"], index=2, horizontal=True,
    )
    historical_default_map = {"No previous default": "N", "Previous default": "Y", "Not reported": "Unknown"}
    st.markdown('</div>', unsafe_allow_html=True)

    raw_input = {
        "age": age, "income": income, "home_ownership": home_ownership,
        "employment_duration": employment_duration, "historical_default": historical_default_map[historical_default_choice],
        "cred_hist_length": cred_hist_length, "loan_amnt": loan_amnt, "loan_int_rate": loan_int_rate,
        "term_years": term_years, "loan_intent": loan_intent, "loan_grade": loan_grade,
    }

    st.markdown('<div class="section-title">⚡ Risk Assessment</div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1, 1])
    with center:
        assess_clicked = st.button("Assess Loan Risk", use_container_width=True)

    if not assess_clicked:
        st.info("Set the customer and loan details in the sidebar, then click **Assess Loan Risk**.")
        return

    try:
        input_df = prepare_input(raw_input)
        prediction, default_proba, proba, classes = predict_risk(input_df, model, encoder)
    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}")
        return

    risk_score, risk_level, risk_color = calculate_risk_score(default_proba)
    df_ref = load_reference_data()

    display_prediction(prediction, default_proba, risk_score, risk_level)

    st.markdown('<div class="section-title">Risk Score</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        display_risk_gauge(risk_score, risk_color)
    with g2:
        display_probability_chart(default_proba)

    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
    display_kpi_cards(raw_input)

    st.markdown('<div class="section-title">Repayment Timeline</div>', unsafe_allow_html=True)
    display_amortization_chart(raw_input)

    st.markdown('<div class="section-title">Why This Prediction?</div>', unsafe_allow_html=True)
    display_feature_importance(tree_model, encoder)

    display_profile_radar(raw_input, df_ref)
    display_loan_vs_income(raw_input, df_ref)

    display_model_info()
    display_tree_explorer()


if __name__ == "__main__":
    main()
