"""Page 1 — KPI overview."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from _shared import (
    apply_theme, load_clean, page_footer, section,
    DATA_DICTIONARY, TEAL, RED, INK,
)

apply_theme()
st.title("Overview")
df = load_clean()

n = len(df)
churn_rate = df["churn_binary"].mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{n:,}")
c2.metric("Churn rate", f"{churn_rate:.2f}%")
c3.metric("Avg monthly charges", f"${df['MonthlyCharges'].mean():.2f}")
c4.metric("Avg tenure", f"{df['tenure'].mean():.1f} months")

section("Churned vs retained")
left, right = st.columns([1, 1])
with left:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    retained = (df["churn_binary"] == 0).sum()
    churned = (df["churn_binary"] == 1).sum()
    ax.pie([retained, churned], colors=[TEAL, RED], startangle=90,
           wedgeprops=dict(width=0.45, edgecolor="white"))
    ax.text(0, 0, f"{churn_rate:.1f}%\nchurn", ha="center", va="center",
            fontsize=15, color=INK, weight="bold")
    fig.patch.set_facecolor("white")
    st.pyplot(fig)
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"**{churned:,} customers left.** That's more than one in four. "
        "The rest of this dashboard is about finding them *before* they go."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">The single biggest split: contract type. '
        "Month-to-month customers churn at 42.7%; two-year contracts at 2.8%.</div>",
        unsafe_allow_html=True,
    )

section("Highest-churn segments")
seg = (df.groupby(["Contract", "InternetService"], observed=True)["churn_binary"]
         .agg(["mean", "size"]).reset_index())
seg = seg[seg["size"] >= 50].sort_values("mean", ascending=False).head(5)
seg["churn rate"] = (seg["mean"] * 100).round(1).astype(str) + "%"
seg = seg.rename(columns={"Contract": "Contract", "InternetService": "Internet",
                           "size": "Customers"})[["Contract", "Internet", "churn rate", "Customers"]]
st.dataframe(seg, use_container_width=True, hide_index=True)
st.caption("Segments with at least 50 customers, ranked by churn rate.")

with st.expander(f"Data dictionary ({len(DATA_DICTIONARY)} columns)"):
    for col, desc in DATA_DICTIONARY.items():
        st.markdown(f"**{col}** — {desc}")

page_footer()
