"""Page 2 — interactive exploratory analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from _shared import apply_theme, load_clean, page_footer, section, ROOT, TEAL, RED, AMBER

apply_theme()
st.title("Explore the data")
df = load_clean()


def internet_label(v: str) -> str:
    return "No internet service" if v == "No" else v


st.sidebar.header("Filters")
contracts = st.sidebar.multiselect(
    "Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
internets = st.sidebar.multiselect(
    "Internet service", sorted(df["InternetService"].unique()),
    default=sorted(df["InternetService"].unique()), format_func=internet_label)
bands = st.sidebar.multiselect(
    "Tenure band (months)", ["0-6", "7-12", "13-24", "25+"],
    default=["0-6", "7-12", "13-24", "25+"])

fdf = df[df["Contract"].isin(contracts)
         & df["InternetService"].isin(internets)
         & df["tenure_band"].isin(bands)]
st.caption(f"Showing {len(fdf):,} of {len(df):,} customers after filters.")
if fdf.empty:
    st.warning("No customers match the selected filters — pick at least one option per filter.")
    st.stop()


def bar(s, title, color):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s.index.astype(str), s.values, color=color)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    for i, v in enumerate(s.values):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    st.pyplot(fig)


section("Churn rate by contract")
s = fdf.groupby("Contract", observed=True)["churn_binary"].mean().sort_values() * 100
colors = [TEAL if v < 15 else AMBER if v < 30 else RED for v in s.values]
bar(s, "Churn rate by contract type", colors)
top = s.idxmax()
st.markdown(
    f'<div class="callout">In this view, <b>{top}</b> contracts churn the most '
    f"({s.max():.1f}%). Longer commitment, lower churn — that pattern holds "
    "across every filter combination.</div>",
    unsafe_allow_html=True,
)

section("Churn rate by tenure band")
order = [b for b in ["0-6", "7-12", "13-24", "25+"] if b in fdf["tenure_band"].unique()]
s = fdf.groupby("tenure_band", observed=True)["churn_binary"].mean().reindex(order) * 100
colors = [RED if i == 0 else AMBER if i == 1 else TEAL for i in range(len(s))]
bar(s, "Churn rate by tenure band (months)", colors)
st.markdown(
    f'<div class="callout">New customers are the risk: the {s.idxmax()} band '
    f"churns at {s.max():.1f}%, falling to {s.min():.1f}% for {s.idxmin()}. "
    "Onboarding is where retention is won or lost.</div>",
    unsafe_allow_html=True,
)

section("Monthly charges: churned vs retained")
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(fdf.loc[fdf["churn_binary"] == 0, "MonthlyCharges"], bins=30,
        alpha=0.65, label="Retained", color=TEAL)
ax.hist(fdf.loc[fdf["churn_binary"] == 1, "MonthlyCharges"], bins=30,
        alpha=0.65, label="Churned", color=RED)
ax.set_xlabel("Monthly charges ($)")
ax.set_ylabel("Customers")
ax.set_title("Monthly charges distribution")
ax.legend()
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
st.pyplot(fig)
gap = (fdf.loc[fdf["churn_binary"] == 1, "MonthlyCharges"].mean()
       - fdf.loc[fdf["churn_binary"] == 0, "MonthlyCharges"].mean())
st.markdown(
    f'<div class="callout">Churned customers pay <b>${gap:.2f}/mo more</b> on '
    "average in this view. Higher bills, higher exit risk.</div>",
    unsafe_allow_html=True,
)

heatmap = ROOT / "figures" / "correlation_heatmap.png"
if heatmap.exists():
    section("Feature correlations")
    st.image(str(heatmap))
    st.caption("Month-to-month contracts, short tenure, and fiber are the top churn associates.")

page_footer()
