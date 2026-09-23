"""Streamlit landing page for the churn analytics project."""
import streamlit as st

from _shared import (
    configure_page,
    load_clean,
    load_metrics,
    model_status,
    page_footer,
    sidebar_project_info,
)

configure_page("Home")
sidebar_project_info()

df = load_clean()
metrics = load_metrics()
churn_rate = float(df["churn_binary"].mean())
cutoff = float(metrics["threshold_selection"]["value"])

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow" style="color:#99F6E4;">End-to-end analytics project</div>'
    '<h1>Telco Customer Churn Analytics</h1>'
    '<p>From raw customer data to explainable churn risk: reproducible cleaning, EDA, '
    'an interpretable model, and a deployable Streamlit decision dashboard.</p>'
    '<div class="hero-pills">'
    f'<span class="hero-pill">{len(df):,} customers</span>'
    f'<span class="hero-pill">{churn_rate:.1%} overall churn</span>'
    f'<span class="hero-pill">ROC-AUC {metrics["roc_auc"]:.3f}</span>'
    f'<span class="hero-pill">Operating cutoff {cutoff:.0%}</span>'
    '<span class="hero-pill">No test-set threshold leakage</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers analyzed", f"{len(df):,}")
c2.metric("Overall churn", f"{churn_rate:.1%}")
c3.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
c4.metric("Operating recall", f"{metrics['test_operating']['recall']:.1%}")

healthy, health_message = model_status()
if healthy:
    st.success(f"Project health check passed: {health_message}", icon="✅")
else:
    st.error(f"Project health check failed: {health_message}", icon="🚨")

st.markdown("### Explore the project")
cols = st.columns(4)
with cols[0]:
    st.markdown(
        '<div class="card navcard"><div class="eyebrow">01 · Overview</div>'
        '<b>What is happening?</b><br><span class="small-muted">KPIs, churn composition, '
        'and the highest-risk customer segments.</span></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Overview.py", label="Open overview →")
with cols[1]:
    st.markdown(
        '<div class="card navcard"><div class="eyebrow">02 · Explore</div>'
        '<b>Where is churn concentrated?</b><br><span class="small-muted">Filter the cohort '
        'and inspect contract, tenure, service, and billing patterns.</span></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Explore.py", label="Open explorer →")
with cols[2]:
    st.markdown(
        '<div class="card navcard"><div class="eyebrow">03 · Predictor</div>'
        '<b>Who needs attention?</b><br><span class="small-muted">Score a logically valid '
        'customer profile, explain the score, and test retention scenarios.</span></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Predictor.py", label="Open predictor →")
with cols[3]:
    st.markdown(
        '<div class="card navcard"><div class="eyebrow">04 · Model & report</div>'
        '<b>Can the evidence be defended?</b><br><span class="small-muted">Untouched-test metrics, '
        'validation-only threshold selection, model effects, and limitations.</span></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Model_and_Report.py", label="Open model report →")

st.markdown("### Pipeline")
st.markdown(
    "**Raw IBM CSV** → schema/data-quality checks → **clean + engineered analysis features** → "
    "reproducible EDA → **60/20/20 stratified train/validation/test split** → logistic-regression "
    "pipeline → validation-only operating threshold → untouched test evaluation → **serialized model + dashboard**."
)

with st.expander("What makes this version submission-ready?"):
    st.markdown(
        "- The predictor and training code share one feature-building contract.\n"
        "- Impossible phone/internet-service combinations are prevented in the UI.\n"
        "- Exact duplicate indicator features were removed from modeling.\n"
        "- One-hot encoding uses reference categories, making coefficient interpretation defensible.\n"
        "- The operating threshold is selected on validation data, not the test set.\n"
        "- Paths are repository-relative and deployment artifacts are included.\n"
        "- `python src/04_validate.py` checks the data/model/UI feature wiring from a fresh clone."
    )

page_footer()
