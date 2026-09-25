"""Overview page: headline KPIs and highest-risk segments."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from _shared import (
    DATA_DICTIONARY,
    INK,
    RED,
    TEAL,
    configure_page,
    load_clean,
    page_footer,
    section,
    sidebar_project_info,
)

configure_page("Overview")
sidebar_project_info()
st.title("Overview")
st.caption("The business picture before any modeling assumptions.")

df = load_clean()
n = len(df)
churned = int(df["churn_binary"].sum())
retained = n - churned
churn_rate = churned / n

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{n:,}")
c2.metric("Churned", f"{churned:,}", f"{churn_rate:.1%} of base")
c3.metric("Avg monthly charges", f"${df['MonthlyCharges'].mean():.2f}")
c4.metric("Avg tenure", f"{df['tenure'].mean():.1f} months")

section("Churn composition")
left, right = st.columns([1, 1.2])
with left:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.pie(
        [retained, churned],
        colors=[TEAL, RED],
        startangle=90,
        wedgeprops={"width": 0.43, "edgecolor": "white"},
    )
    ax.text(0, 0, f"{churn_rate:.1%}\nchurn", ha="center", va="center",
            fontsize=16, color=INK, weight="bold")
    st.pyplot(fig, clear_figure=True)
with right:
    contract = (
        df.groupby("Contract", observed=True)["churn_binary"]
        .mean()
        .reindex(["Month-to-month", "One year", "Two year"])
        * 100
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    colors = [RED, "#D97706", TEAL]
    bars = ax.bar(contract.index, contract.values, color=colors)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Contract type is the clearest segment split", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, contract.values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 1, f"{value:.1f}%",
                ha="center", fontsize=9)
    st.pyplot(fig, clear_figure=True)

st.markdown(
    '<div class="callout"><b>Headline:</b> month-to-month customers churn at '
    f'{contract["Month-to-month"]:.1f}%, compared with {contract["One year"]:.1f}% on one-year '
    f'contracts and {contract["Two year"]:.1f}% on two-year contracts. '
    'This is an association, not proof that changing a contract alone causes retention.</div>',
    unsafe_allow_html=True,
)

section("Highest-churn segments")
segments = (
    df.groupby(["Contract", "InternetService"], observed=True)["churn_binary"]
    .agg(churn_rate="mean", customers="size")
    .reset_index()
)
segments = segments[segments["customers"] >= 50].sort_values(
    ["churn_rate", "customers"], ascending=[False, False]
).head(8)
segments["Churn rate"] = (segments["churn_rate"] * 100).map(lambda x: f"{x:.1f}%")
segments = segments.rename(
    columns={"Contract": "Contract", "InternetService": "Internet", "customers": "Customers"}
)
st.dataframe(
    segments[["Contract", "Internet", "Customers", "Churn rate"]],
    width="stretch",
    hide_index=True,
)
st.caption("Only segment combinations with at least 50 customers are shown.")

tenure_rates = df.groupby("tenure_band", observed=True)["churn_binary"].mean() * 100
internet_rates = df.groupby("InternetService", observed=True)["churn_binary"].mean() * 100
payment_rates = df.groupby("PaymentMethod", observed=True)["churn_binary"].mean() * 100
support_rates = df.groupby("TechSupport", observed=True)["churn_binary"].mean() * 100

section("Four observations to remember")
obs1, obs2 = st.columns(2)
with obs1:
    st.markdown(
        '<div class="card"><b>New-customer risk</b><br><span class="small-muted">'
        f'Customers in months 0-6 churn at {tenure_rates["0-6"]:.1f}%; customers at 25+ months churn at {tenure_rates["25+"]:.1f}%.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card"><b>Fiber segment</b><br><span class="small-muted">'
        f'Fiber-optic customers churn at {internet_rates["Fiber optic"]:.1f}%, versus {internet_rates["DSL"]:.1f}% for DSL customers.</span></div>',
        unsafe_allow_html=True,
    )
with obs2:
    st.markdown(
        '<div class="card"><b>Payment pattern</b><br><span class="small-muted">'
        f'Electronic-check customers churn at {payment_rates["Electronic check"]:.1f}%; this is a targeting clue, not a causal conclusion.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card"><b>Support pattern</b><br><span class="small-muted">'
        f'Customers without tech support churn at {support_rates["No"]:.1f}% versus {support_rates["Yes"]:.1f}% among customers with support.</span></div>',
        unsafe_allow_html=True,
    )

with st.expander(f"Data dictionary ({len(DATA_DICTIONARY)} columns after engineering)"):
    for column, description in DATA_DICTIONARY.items():
        st.markdown(f"**{column}** — {description}")

page_footer()
