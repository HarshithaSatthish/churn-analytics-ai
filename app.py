"""Landing page — hero, navigation cards, footer."""
import streamlit as st
from _shared import apply_theme, load_clean, load_metrics, page_footer

st.set_page_config(page_title="Telco Churn Analytics", layout="wide")
apply_theme()

df = load_clean()
m = load_metrics()
churn_rate = df["churn_binary"].mean() * 100

st.markdown(
    '<div class="hero">'
    "<h1>Telco Customer Churn Analytics</h1>"
    "<p>7,043 customers. One question: who is about to leave, and why?</p>"
    "<p>Cleaned data, an honest analysis, a trained churn model, and a live "
    "predictor — all built in this project.</p>"
    "</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Customers analyzed", f"{len(df):,}")
c2.metric("Overall churn rate", f"{churn_rate:.1f}%")
c3.metric("Model ROC-AUC", f"{m['roc_auc']:.3f}")

st.markdown("### Start here")
cols = st.columns(4)
with cols[0]:
    st.markdown('<div class="navcard"><b>Overview</b><br>The headline numbers and '
                "worst segments at a glance.</div>", unsafe_allow_html=True)
    st.page_link("pages/1_Overview.py", label="Open Overview")
with cols[1]:
    st.markdown('<div class="navcard"><b>Explore</b><br>Slice the data by contract, '
                "internet service, and tenure.</div>", unsafe_allow_html=True)
    st.page_link("pages/2_Explore.py", label="Open Explore")
with cols[2]:
    st.markdown('<div class="navcard"><b>Predictor</b><br>Type in a customer profile, '
                "get a live churn score.</div>", unsafe_allow_html=True)
    st.page_link("pages/3_Predictor.py", label="Open Predictor")
with cols[3]:
    st.markdown('<div class="navcard"><b>Model &amp; Report</b><br>How the model was '
                "built, and what it found.</div>", unsafe_allow_html=True)
    st.page_link("pages/4_Model_and_Report.py", label="Open Model & Report")

st.markdown("### How to use this dashboard")
st.markdown(
    "1. **Overview** gives you the big picture in under a minute.\n"
    "2. **Explore** lets you test your own hunches with filters.\n"
    "3. **Predictor** scores any customer profile with the trained model — "
    "including why it scored that way.\n"
    "4. **Model & Report** shows the evidence: metrics, drivers, findings."
)

with st.expander("About this project"):
    st.markdown(
        "Built for the Data Analytics with AI final project. Pipeline: raw IBM "
        "Telco Customer Churn data -> cleaning and feature engineering -> "
        "exploratory analysis -> logistic-regression model (80/20 split, "
        "balanced class weights) -> this dashboard. The model and every number "
        "here come from the same run; nothing is hand-edited."
    )

page_footer()
