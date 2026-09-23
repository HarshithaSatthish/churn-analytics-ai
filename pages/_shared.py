"""Shared helpers for dashboard pages.

Each page is standalone (Streamlit multipage runs pages as separate scripts),
so loaders, the theme, and scoring helpers live here. Paths resolve from this
file's location.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- design system
TEAL = "#0E7C7B"
AMBER = "#D97706"
RED = "#B91C1C"
GREEN = "#15803D"
INK = "#1C2430"
MUTED = "#6B7280"
PAPER = "#F7F4EE"
CARD_BORDER = "#E7E1D3"

THEME_CSS = f"""
<style>
.block-container {{ max-width: 1120px; padding-top: 1.6rem; }}
[data-testid="stMetric"] {{
  background: #ffffff; border: 1px solid {CARD_BORDER}; border-radius: 14px;
  padding: 12px 16px; box-shadow: 0 1px 4px rgba(28,36,48,.07);
}}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
.card {{
  background: #ffffff; border: 1px solid {CARD_BORDER}; border-radius: 14px;
  padding: 1.0rem 1.2rem; margin: .6rem 0;
  box-shadow: 0 1px 4px rgba(28,36,48,.06);
}}
.callout {{
  border-left: 4px solid {TEAL}; background: #ffffff;
  padding: .7rem 1rem; border-radius: 0 10px 10px 0; margin: .6rem 0;
}}
.sec {{ display: flex; align-items: stretch; gap: .7rem; margin: 1.5rem 0 .7rem; }}
.sec-bar {{ width: 6px; border-radius: 3px; background: {TEAL}; flex: none; }}
.sec h3 {{ margin: 0; padding: .1rem 0; }}
.band {{ border-radius: 12px; padding: .85rem 1.1rem; margin: .6rem 0; color: #ffffff; }}
.band-high {{ background: {RED}; }}
.band-med {{ background: #B45309; }}
.band-low {{ background: {TEAL}; }}
.gauge {{
  position: relative; height: 14px; border-radius: 7px;
  background: linear-gradient(90deg, {TEAL} 0 30%, {AMBER} 30% 60%, {RED} 60% 100%);
  margin: .4rem 0 1.4rem;
}}
.gauge-marker {{
  position: absolute; top: -6px; width: 4px; height: 26px;
  background: {INK}; border-radius: 2px; transform: translateX(-2px);
}}
.gauge-labels {{ display: flex; justify-content: space-between; color: {MUTED};
  font-size: .8rem; margin-top: -1rem; margin-bottom: .6rem; }}
.stButton > button {{ border-radius: 10px; }}
.stDownloadButton > button {{ border-radius: 10px; }}
footer {{ visibility: hidden; }}
.hero {{
  background: linear-gradient(135deg, #123B3B 0%, #0E7C7B 70%, #159186 100%);
  border-radius: 18px; padding: 2rem 2.2rem; color: #ffffff; margin-bottom: 1.2rem;
}}
.hero h1 {{ color: #ffffff; margin: 0 0 .4rem; }}
.hero p {{ color: #E8F3F2; margin: .2rem 0; }}
.navcard {{
  background: #ffffff; border: 1px solid {CARD_BORDER}; border-radius: 14px;
  padding: 1rem 1.2rem; margin: .4rem 0; box-shadow: 0 1px 4px rgba(28,36,48,.06);
}}
.page-footer {{ color: {MUTED}; font-size: .85rem; margin-top: 2.5rem;
  padding-top: 1rem; border-top: 1px solid {CARD_BORDER}; }}
.contrib-row {{ display: flex; align-items: center; gap: .6rem; margin: .3rem 0; }}
.contrib-label {{ flex: 0 0 46%; font-size: .9rem; }}
.contrib-bar-wrap {{ flex: 1; background: #F1ECE0; border-radius: 6px; height: 12px; }}
.contrib-bar {{ height: 12px; border-radius: 6px; }}
.contrib-val {{ flex: 0 0 64px; text-align: right; font-size: .85rem; color: {MUTED}; }}
</style>
"""


def apply_theme() -> None:
    """Inject the design-system CSS. Call once at the top of every page."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(
        f'<div class="sec"><div class="sec-bar"></div><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


def page_footer() -> None:
    st.markdown(
        '<div class="page-footer">Telco Customer Churn Analytics · data: IBM '
        'Telco Customer Churn (Apache-2.0) · model: logistic regression trained '
        'on this data · built with Streamlit</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------- loaders
@st.cache_data
def load_clean() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "clean" / "telco_clean.csv")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((ROOT / "models" / "metrics.json").read_text())


@st.cache_data
def load_feature_info() -> dict:
    return json.loads((ROOT / "models" / "feature_info.json").read_text())


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "churn_model.pkl")


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    """Contract file from the backend team; None until it lands."""
    p = ROOT / "models" / "feature_importance.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def cohort_scores() -> np.ndarray:
    """Churn probability for every customer in the clean dataset."""
    df = load_clean()
    info = load_feature_info()
    model = load_model()
    X = df[info["categorical"] + info["numeric"]]
    return model.predict_proba(X)[:, 1]


# ------------------------------------------------- feature labels and reasons
FEATURE_DESCRIPTIONS = {
    "Contract_Month-to-month": "Month-to-month contract",
    "Contract_One year": "One-year contract",
    "Contract_Two year": "Two-year contract",
    "tenure_band_0-6": "Customer for 6 months or less",
    "tenure_band_7-12": "Customer for 7-12 months",
    "tenure_band_13-24": "Customer for 13-24 months",
    "tenure_band_25+": "Customer for over 2 years",
    "InternetService_Fiber optic": "Fiber-optic internet",
    "InternetService_DSL": "DSL internet",
    "InternetService_No": "No internet service",
    "PaymentMethod_Electronic check": "Pays by electronic check",
    "PaymentMethod_Mailed check": "Pays by mailed check",
    "PaymentMethod_Bank transfer (automatic)": "Pays by bank transfer",
    "PaymentMethod_Credit card (automatic)": "Pays by credit card",
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges to date",
    "num_services": "Number of add-on services",
    "is_fiber": "Fiber-optic flag",
    "is_month_to_month": "Month-to-month flag",
    "is_electronic_check": "Electronic-check flag",
    "SeniorCitizen": "Senior citizen",
    "TechSupport_No": "No tech support",
    "TechSupport_Yes": "Has tech support",
    "OnlineSecurity_No": "No online security",
    "OnlineSecurity_Yes": "Has online security",
    "gender_Male": "Male customer",
    "gender_Female": "Female customer",
    "Partner_Yes": "Has a partner",
    "Partner_No": "No partner",
    "Dependents_Yes": "Has dependents",
    "Dependents_No": "No dependents",
    "PaperlessBilling_Yes": "Paperless billing",
    "PaperlessBilling_No": "Paper billing",
    "PhoneService_Yes": "Has phone service",
    "PhoneService_No": "No phone service",
    "MultipleLines_Yes": "Multiple phone lines",
    "StreamingTV_Yes": "Streams TV",
    "StreamingMovies_Yes": "Streams movies",
    "OnlineBackup_Yes": "Has online backup",
    "DeviceProtection_Yes": "Has device protection",
}


def feature_label(raw: str) -> str:
    """Human label for a one-hot/numeric model feature."""
    if raw in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[raw]
    return raw.replace("_", " ")


def contribution_breakdown(pipe, X_row: pd.DataFrame, top_pos: int = 3,
                           top_neg: int = 2):
    """Per-prediction reason codes for the logistic-regression model.

    For a linear model, contribution = coefficient * transformed feature value
    is exact — this is the same quantity a SHAP linear explainer returns, with
    no extra dependency. Returns (drivers, protectors) as lists of
    (label, contribution) tuples.
    """
    pre = pipe.named_steps["preprocess"]
    coefs = pipe.named_steps["clf"].coef_[0]
    Xt = pre.transform(X_row)[0]
    contrib = Xt * coefs
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    order = np.argsort(contrib)
    protectors = [(feature_label(names[i]), float(contrib[i]))
                  for i in order[:top_neg] if contrib[i] < 0]
    drivers = [(feature_label(names[i]), float(contrib[i]))
               for i in order[::-1][:top_pos] if contrib[i] > 0]
    return drivers, protectors


def contribution_bars(items, color: str) -> None:
    """Render (label, value) pairs as simple horizontal bars."""
    if not items:
        st.caption("None strong enough to list.")
        return
    scale = max(abs(v) for _, v in items) or 1.0
    for label, val in items:
        width = abs(val) / scale * 100
        st.markdown(
            f'<div class="contrib-row"><div class="contrib-label">{label}</div>'
            f'<div class="contrib-bar-wrap"><div class="contrib-bar" '
            f'style="width:{width:.1f}%;background:{color};"></div></div>'
            f'<div class="contrib-val">{val:+.2f}</div></div>',
            unsafe_allow_html=True,
        )


def odds_ratio_table(pipe, top_n: int = 12) -> pd.DataFrame:
    """Feature -> coef -> odds ratio, sorted by |coef|. From the live model,
    identical to the backend's feature_importance.csv when present."""
    pre = pipe.named_steps["preprocess"]
    coefs = pipe.named_steps["clf"].coef_[0]
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    df = pd.DataFrame({
        "feature": [feature_label(n) for n in names],
        "coef": coefs,
        "odds_ratio": np.exp(coefs),
    })
    return df.reindex(df["coef"].abs().sort_values(ascending=False).index).head(top_n)


def plain_interpretation(row) -> str:
    """One plain-English line for an odds-ratio table row."""
    ora = row["odds_ratio"]
    if row["coef"] > 0:
        return f"Churn odds about {ora:.1f}x higher than the average customer."
    return f"Churn odds about {1 / ora:.1f}x lower than the average customer."


def risk_band(proba: float):
    """(band name, css class, recommended action). Thresholds shown in-page."""
    if proba < 0.30:
        return ("LOW", "band-low",
                "No action needed. Standard billing and service is enough.")
    if proba < 0.60:
        return ("MEDIUM", "band-med",
                "Worth a proactive touch: a check-in call or a small loyalty offer.")
    return ("HIGH", "band-high",
            "Prioritize for the retention team now: contract-upgrade or save offer.")


def gauge(proba: float) -> None:
    """Visual risk gauge: green -> amber -> red zones, marker at the score."""
    pct = max(0.0, min(100.0, proba * 100))
    st.markdown(
        f'<div class="gauge"><div class="gauge-marker" style="left:{pct:.1f}%;"></div></div>'
        f'<div class="gauge-labels"><span>0% · low</span>'
        f'<span>30%</span><span>60%</span><span>100% · high</span></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------- predictor row builder
def profile_to_row(p: dict) -> pd.DataFrame:
    """Raw profile dict -> one-row DataFrame with the exact 24 model features.

    Keys: gender, senior(bool), partner, dependents, tenure(int),
    phone, multi, internet, onlinesec, onlinebak, devprot, techsup, tv, movies,
    contract, paperless, paymethod, monthly(float).
    """
    tenure = int(p["tenure"])
    monthly = float(p["monthly"])
    svc_vals = [p["phone"], p["multi"], p["internet"], p["onlinesec"],
                p["onlinebak"], p["devprot"], p["techsup"], p["tv"], p["movies"]]
    tenure_band = ("0-6" if tenure <= 6 else "7-12" if tenure <= 12
                   else "13-24" if tenure <= 24 else "25+")
    row = {
        "gender": p["gender"], "Partner": p["partner"], "Dependents": p["dependents"],
        "PhoneService": p["phone"], "MultipleLines": p["multi"],
        "InternetService": p["internet"], "OnlineSecurity": p["onlinesec"],
        "OnlineBackup": p["onlinebak"], "DeviceProtection": p["devprot"],
        "TechSupport": p["techsup"], "StreamingTV": p["tv"],
        "StreamingMovies": p["movies"], "Contract": p["contract"],
        "PaperlessBilling": p["paperless"], "PaymentMethod": p["paymethod"],
        "tenure_band": tenure_band,
        "SeniorCitizen": int(p["senior"]), "tenure": tenure,
        "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
        "num_services": sum(1 for v in svc_vals if v == "Yes"),
        "is_fiber": int(p["internet"] == "Fiber optic"),
        "is_month_to_month": int(p["contract"] == "Month-to-month"),
        "is_electronic_check": int(p["paymethod"] == "Electronic check"),
    }
    info = load_feature_info()
    return pd.DataFrame([row])[info["categorical"] + info["numeric"]]


# ------------------------------------------------- threshold point metrics
@st.cache_data
def threshold_point_metrics(threshold: float) -> dict:
    """Precision/recall/F1 on the held-out test split at a given cutoff.

    Rebuilds the exact 80/20 stratified split (random_state=42) the model was
    evaluated on — deterministic, no retraining, no new dependencies.
    """
    from sklearn.model_selection import train_test_split
    df = load_clean()
    info = load_feature_info()
    X = df[info["categorical"] + info["numeric"]]
    y = df[info["target"]]
    _, Xte, _, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    proba = load_model().predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    yv = yte.values
    tp = int(((pred == 1) & (yv == 1)).sum())
    fp = int(((pred == 1) & (yv == 0)).sum())
    fn = int(((pred == 0) & (yv == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(threshold), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


DATA_DICTIONARY = {
    "customerID": "Unique customer identifier",
    "gender": "Male / Female",
    "SeniorCitizen": "1 if senior citizen, else 0",
    "Partner": "Has a partner (Yes/No)",
    "Dependents": "Has dependents (Yes/No)",
    "tenure": "Months with the company",
    "PhoneService": "Has phone service (Yes/No)",
    "MultipleLines": "Has multiple lines (Yes/No/No phone service)",
    "InternetService": "DSL / Fiber optic / No (= no internet service)",
    "OnlineSecurity": "Has online security (Yes/No/No internet service)",
    "OnlineBackup": "Has online backup (Yes/No/No internet service)",
    "DeviceProtection": "Has device protection (Yes/No/No internet service)",
    "TechSupport": "Has tech support (Yes/No/No internet service)",
    "StreamingTV": "Streams TV (Yes/No/No internet service)",
    "StreamingMovies": "Streams movies (Yes/No/No internet service)",
    "Contract": "Month-to-month / One year / Two year",
    "PaperlessBilling": "Paperless billing (Yes/No)",
    "PaymentMethod": "Electronic check / Mailed check / Bank transfer / Credit card",
    "MonthlyCharges": "Monthly bill amount ($)",
    "TotalCharges": "Total billed to date ($)",
    "Churn": "Target: Yes if the customer left",
    # Engineered features (added during cleaning)
    "tenure_band": "Engineered: tenure grouped as 0-6 / 7-12 / 13-24 / 25+ months",
    "num_services": "Engineered: count of Yes across the 9 service columns",
    "is_fiber": "Engineered: 1 if InternetService is Fiber optic, else 0",
    "is_month_to_month": "Engineered: 1 if Contract is Month-to-month, else 0",
    "is_electronic_check": "Engineered: 1 if PaymentMethod is Electronic check, else 0",
    "churn_binary": "Engineered target: 1 if Churn is Yes, else 0",
}
