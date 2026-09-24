"""Shared data, model, and presentation helpers for the Streamlit dashboard."""
from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import profile_to_model_row
from src.utils import RISK_LOW, RISK_VERY_HIGH

ROOT = Path(__file__).resolve().parent

TEAL = "#0F766E"
TEAL_DARK = "#115E59"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#B91C1C"
PURPLE = "#7C3AED"
INK = "#0F172A"
MUTED = "#64748B"
PAPER = "#F8FAFC"
CARD_BORDER = "#E2E8F0"

THEME_CSS = f"""
<style>
html, body, [class*="css"] {{ color: {INK}; }}
.block-container {{ max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {CARD_BORDER}; }}
[data-testid="stSidebarNav"] {{ display: none; }}
[data-testid="stMetric"] {{
  background: white; border: 1px solid {CARD_BORDER}; border-radius: 16px;
  padding: 14px 16px; box-shadow: 0 8px 24px rgba(15,23,42,.045);
}}
[data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: {INK}; }}
.hero {{
  background: linear-gradient(135deg, #0F172A 0%, #134E4A 55%, #0F766E 100%);
  border-radius: 22px; padding: 2.2rem 2.35rem; color: white; margin: .2rem 0 1.25rem;
  box-shadow: 0 18px 42px rgba(15,23,42,.14);
}}
.hero h1 {{ color: white; font-size: 2.25rem; margin: 0 0 .45rem; line-height: 1.1; }}
.hero p {{ color: #D5F5F1; font-size: 1.02rem; margin: .25rem 0; max-width: 820px; }}
.hero-pills {{ display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1rem; }}
.hero-pill {{ background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
  color: #F8FAFC; padding: .35rem .62rem; border-radius: 999px; font-size: .82rem; }}
.card {{
  background: white; border: 1px solid {CARD_BORDER}; border-radius: 16px;
  padding: 1.05rem 1.2rem; margin: .5rem 0; box-shadow: 0 7px 22px rgba(15,23,42,.04);
}}
.navcard {{ min-height: 150px; }}
.eyebrow {{ color: {TEAL}; text-transform: uppercase; letter-spacing: .08em;
  font-size: .74rem; font-weight: 800; margin-bottom: .25rem; }}
.callout {{ border-left: 4px solid {TEAL}; background: #F0FDFA; padding: .8rem 1rem;
  border-radius: 0 12px 12px 0; margin: .65rem 0; }}
.callout-warn {{ border-left-color: {AMBER}; background: #FFFBEB; }}
.callout-danger {{ border-left-color: {RED}; background: #FEF2F2; }}
.sec {{ display: flex; align-items: center; gap: .7rem; margin: 1.55rem 0 .7rem; }}
.sec-bar {{ width: 6px; height: 29px; border-radius: 4px; background: {TEAL}; }}
.sec h3 {{ margin: 0; padding: 0; font-size: 1.25rem; }}
.band {{ border-radius: 14px; padding: .9rem 1.05rem; margin: .65rem 0; color: white; }}
.band-low {{ background: {TEAL}; }}
.band-medium {{ background: {AMBER}; }}
.band-high {{ background: {RED}; }}
.band-very-high {{ background: #7F1D1D; }}
.gauge {{ position: relative; height: 15px; border-radius: 999px; margin: .65rem 0 1.5rem;
  background: linear-gradient(90deg, {TEAL} 0 20%, {AMBER} 20% 36%, {RED} 36% 60%, #7F1D1D 60% 100%); }}
.gauge-marker {{ position: absolute; top: -7px; width: 4px; height: 29px; background: {INK};
  border-radius: 4px; transform: translateX(-2px); box-shadow: 0 0 0 2px white; }}
.gauge-labels {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .78rem;
  margin-top: -1rem; }}
.contrib-row {{ display: grid; grid-template-columns: minmax(140px, 1.5fr) 2fr 58px;
  gap: .55rem; align-items: center; margin: .4rem 0; }}
.contrib-label {{ font-size: .88rem; }}
.contrib-wrap {{ background: #EEF2F7; border-radius: 999px; height: 11px; overflow: hidden; }}
.contrib-bar {{ height: 11px; border-radius: 999px; }}
.contrib-val {{ text-align: right; font-size: .82rem; color: {MUTED}; }}
.status-ok {{ color: {TEAL_DARK}; font-weight: 700; }}
.small-muted {{ color: {MUTED}; font-size: .84rem; }}
.page-footer {{ color: {MUTED}; font-size: .82rem; margin-top: 2.5rem; padding-top: 1rem;
  border-top: 1px solid {CARD_BORDER}; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 11px; font-weight: 650; }}
</style>
"""


def configure_page(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} | Churn Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(
        f'<div class="sec"><div class="sec-bar"></div><h3>{html.escape(title)}</h3></div>',
        unsafe_allow_html=True,
    )


def sidebar_project_info() -> None:
    metrics = load_metrics()
    cutoff = metrics["threshold_selection"]["value"]
    with st.sidebar:
        st.markdown("### Churn Analytics")
        st.caption("IBM Telco Customer Churn · 7,043 customers")
        st.markdown(f"**Model:** Logistic regression  \n**Operating cutoff:** {cutoff:.0%}")
        st.caption("Threshold selected on validation data; final metrics use an untouched test set.")
        st.divider()
        st.markdown("**Pages**")
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Overview.py", label="Overview", icon="📌")
        st.page_link("pages/2_Explore.py", label="Explore", icon="🔎")
        st.page_link("pages/3_Predictor.py", label="Predictor", icon="🎯")
        st.page_link("pages/4_Model_and_Report.py", label="Model & Report", icon="🧠")


def page_footer() -> None:
    st.markdown(
        '<div class="page-footer">Telco Customer Churn Analytics · IBM Telco Customer Churn dataset · '
        'interpretable logistic regression · Streamlit dashboard</div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_clean() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "clean" / "telco_clean.csv")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data
def load_feature_info() -> dict:
    return json.loads((ROOT / "models" / "feature_info.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "churn_model.pkl")


@st.cache_data
def load_feature_effects() -> pd.DataFrame:
    return pd.read_csv(ROOT / "models" / "feature_importance.csv")


@st.cache_data
def cohort_scores() -> np.ndarray:
    df = load_clean()
    info = load_feature_info()
    return load_model().predict_proba(df[info["model_features"]])[:, 1]


def model_status() -> tuple[bool, str]:
    try:
        df = load_clean()
        info = load_feature_info()
        model = load_model()
        score = float(model.predict_proba(df[info["model_features"]].iloc[[0]])[0, 1])
        if 0 <= score <= 1:
            return True, "Data, feature schema, and serialized model are connected."
    except Exception as exc:  # UI should show a readable health state, not crash the landing page.
        return False, str(exc)
    return False, "Model returned an invalid score."


FEATURE_LABELS = {
    "gender": "Gender",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "tenure_band": "Tenure band",
    "SeniorCitizen": "Senior citizen",
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges",
    "num_services": "Number of services",
}


def readable_feature(name: str) -> str:
    if "=" in name:
        source, category = name.split("=", 1)
        return f"{FEATURE_LABELS.get(source, source)}: {category}"
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def model_contributions(pipe, X_row: pd.DataFrame, top_pos: int = 4, top_neg: int = 3):
    """Exact additive log-odds contributions for the linear pipeline."""
    pre = pipe.named_steps["preprocess"]
    coefficients = pipe.named_steps["clf"].coef_[0]
    transformed = np.asarray(pre.transform(X_row))[0]
    contributions = transformed * coefficients
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]

    order = np.argsort(contributions)
    protectors = [
        (readable_feature(names[i]), float(contributions[i]))
        for i in order[:top_neg]
        if contributions[i] < 0
    ]
    drivers = [
        (readable_feature(names[i]), float(contributions[i]))
        for i in order[::-1][:top_pos]
        if contributions[i] > 0
    ]
    return drivers, protectors


def contribution_bars(items, color: str) -> None:
    if not items:
        st.caption("No strong contribution to display.")
        return
    scale = max(abs(value) for _, value in items) or 1.0
    for label, value in items:
        width = abs(value) / scale * 100
        st.markdown(
            '<div class="contrib-row">'
            f'<div class="contrib-label">{html.escape(label)}</div>'
            '<div class="contrib-wrap">'
            f'<div class="contrib-bar" style="width:{width:.1f}%;background:{color};"></div>'
            '</div>'
            f'<div class="contrib-val">{value:+.2f}</div>'
            '</div>',
            unsafe_allow_html=True,
        )


def _band_bounds() -> tuple[float, float]:
    """(low_max, very_high_min) from metrics.json, falling back to training constants."""
    try:
        bands = json.loads(
            (ROOT / "models" / "metrics.json").read_text(encoding="utf-8")
        ).get("risk_bands", {})
        return (
            float(bands.get("low_max", RISK_LOW)),
            float(bands.get("very_high_min", RISK_VERY_HIGH)),
        )
    except Exception:
        return RISK_LOW, RISK_VERY_HIGH


def risk_band(score: float, cutoff: float) -> tuple[str, str, str]:
    low_max, very_high_min = _band_bounds()
    if score < low_max:
        return "LOW", "band-low", "Routine service; no retention action indicated."
    if score < cutoff:
        return "WATCH", "band-medium", "Monitor and consider a low-cost proactive check-in."
    if score < very_high_min:
        return "HIGH", "band-high", "Meets the model cutoff; prioritize for retention outreach."
    return "VERY HIGH", "band-very-high", "Top-priority retention case; review contract, support, and offer options."


def gauge(score: float, cutoff: float) -> None:
    low_max, very_high_min = _band_bounds()
    low_pct, vhigh_pct = low_max * 100, very_high_min * 100
    pct = max(0.0, min(100.0, score * 100))
    cutoff_pct = max(low_pct + 0.1, min(vhigh_pct - 0.1, cutoff * 100))
    background = (
        f"linear-gradient(90deg,{TEAL} 0 {low_pct:.1f}%,{AMBER} {low_pct:.1f}% {cutoff_pct:.1f}%,"
        f"{RED} {cutoff_pct:.1f}% {vhigh_pct:.1f}%,#7F1D1D {vhigh_pct:.1f}% 100%)"
    )
    st.markdown(
        f'<div class="gauge" style="background:{background};">'
        f'<div class="gauge-marker" style="left:{pct:.1f}%;"></div></div>'
        '<div class="gauge-labels"><span>0%</span>'
        f'<span>{low_pct:.0f}% watch</span>'
        f'<span>{cutoff_pct:.0f}% cutoff</span>'
        f'<span>{vhigh_pct:.0f}% very high</span><span>100%</span></div>',
        unsafe_allow_html=True,
    )


def profile_to_row(profile: dict) -> pd.DataFrame:
    return profile_to_model_row(profile)


DATA_DICTIONARY = {
    "customerID": "Unique customer identifier",
    "gender": "Customer gender",
    "SeniorCitizen": "1 if senior citizen, otherwise 0",
    "Partner": "Whether the customer has a partner",
    "Dependents": "Whether the customer has dependents",
    "tenure": "Months with the company",
    "PhoneService": "Whether phone service is active",
    "MultipleLines": "Multiple-line status",
    "InternetService": "DSL, Fiber optic, or No internet service",
    "OnlineSecurity": "Online security service status",
    "OnlineBackup": "Online backup service status",
    "DeviceProtection": "Device protection status",
    "TechSupport": "Technical support status",
    "StreamingTV": "Streaming TV status",
    "StreamingMovies": "Streaming movies status",
    "Contract": "Month-to-month, one-year, or two-year contract",
    "PaperlessBilling": "Paperless billing status",
    "PaymentMethod": "Customer payment method",
    "MonthlyCharges": "Current monthly bill amount",
    "TotalCharges": "Total billed amount to date",
    "Churn": "Original Yes/No churn target",
    "tenure_band": "Engineered tenure group: 0-6, 7-12, 13-24, 25+ months",
    "num_services": "Count of service columns whose dataset value is Yes",
    "is_fiber": "Analysis-only indicator for Fiber optic service",
    "is_month_to_month": "Analysis-only indicator for month-to-month contract",
    "is_electronic_check": "Analysis-only indicator for electronic-check payment",
    "churn_binary": "Engineered target: 1 for churn, 0 for retained",
}
