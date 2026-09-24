"""Interactive EDA page with cohort filters and export."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from _shared import (
    AMBER,
    RED,
    TEAL,
    configure_page,
    load_clean,
    page_footer,
    section,
    sidebar_project_info,
)

configure_page("Explore")
sidebar_project_info()
st.title("Explore the data")
st.caption("Change the cohort and see the descriptive churn patterns recalculate immediately.")

df = load_clean()

with st.sidebar:
    st.markdown("### Cohort filters")
    contracts = st.multiselect(
        "Contract",
        options=list(df["Contract"].drop_duplicates()),
        default=list(df["Contract"].drop_duplicates()),
    )
    internets = st.multiselect(
        "Internet service",
        options=list(df["InternetService"].drop_duplicates()),
        default=list(df["InternetService"].drop_duplicates()),
        format_func=lambda value: "No internet" if value == "No" else value,
    )
    tenure_bands = st.multiselect(
        "Tenure band",
        options=["0-6", "7-12", "13-24", "25+"],
        default=["0-6", "7-12", "13-24", "25+"],
    )
    senior_choice = st.radio("Senior citizen", ["All", "No", "Yes"], horizontal=True)

mask = (
    df["Contract"].isin(contracts)
    & df["InternetService"].isin(internets)
    & df["tenure_band"].isin(tenure_bands)
)
if senior_choice != "All":
    mask &= df["SeniorCitizen"].eq(1 if senior_choice == "Yes" else 0)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No customers match this filter combination. Widen at least one filter.")
    st.stop()

base_rate = float(df["churn_binary"].mean())
filtered_rate = float(fdf["churn_binary"].mean())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered customers", f"{len(fdf):,}", f"{len(fdf)/len(df):.1%} of base")
c2.metric("Filtered churn", f"{filtered_rate:.1%}", f"{(filtered_rate-base_rate)*100:+.1f} pts vs all")
c3.metric("Avg monthly bill", f"${fdf['MonthlyCharges'].mean():.2f}")
c4.metric("Avg tenure", f"{fdf['tenure'].mean():.1f} months")

st.download_button(
    "Download filtered cohort (CSV)",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="filtered_churn_cohort.csv",
    mime="text/csv",
)

segments_tab, charges_tab, services_tab, data_tab = st.tabs(
    ["Segments", "Charges & tenure", "Services & payment", "Data"]
)

with segments_tab:
    section("Churn by contract")
    contract = fdf.groupby("Contract", observed=True)["churn_binary"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    colors = [TEAL if value < 15 else AMBER if value < 30 else RED for value in contract]
    bars = ax.bar(contract.index, contract.values, color=colors)
    ax.set_ylabel("Churn rate (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, contract.values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.7, f"{value:.1f}%", ha="center", fontsize=9)
    st.pyplot(fig, clear_figure=True)

    section("Churn by tenure band")
    order = [band for band in ["0-6", "7-12", "13-24", "25+"] if band in set(fdf["tenure_band"])]
    tenure = fdf.groupby("tenure_band", observed=True)["churn_binary"].mean().reindex(order) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    bars = ax.bar(tenure.index, tenure.values, color=[RED, AMBER, "#2563EB", TEAL][:len(tenure)])
    ax.set_ylabel("Churn rate (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, tenure.values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.7, f"{value:.1f}%", ha="center", fontsize=9)
    st.pyplot(fig, clear_figure=True)

with charges_tab:
    section("Monthly charges by outcome")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(fdf.loc[fdf["churn_binary"] == 0, "MonthlyCharges"], bins=28,
            alpha=.66, color=TEAL, label="Retained")
    ax.hist(fdf.loc[fdf["churn_binary"] == 1, "MonthlyCharges"], bins=28,
            alpha=.66, color=RED, label="Churned")
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Customers")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)
    churn_mean = fdf.loc[fdf["churn_binary"] == 1, "MonthlyCharges"].mean()
    keep_mean = fdf.loc[fdf["churn_binary"] == 0, "MonthlyCharges"].mean()
    if pd.notna(churn_mean) and pd.notna(keep_mean):
        st.info(
            f"Within this filtered cohort, churned customers average ${churn_mean:.2f}/month "
            f"versus ${keep_mean:.2f}/month for retained customers."
        )

    section("Tenure distribution")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(fdf["tenure"], bins=30, color="#7C3AED", alpha=.85)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)

with services_tab:
    left, right = st.columns(2)
    with left:
        section("Internet service")
        internet = fdf.groupby("InternetService", observed=True)["churn_binary"].mean().sort_values() * 100
        st.dataframe(
            pd.DataFrame({"Churn rate": internet.map(lambda x: f"{x:.1f}%")}),
            width="stretch",
        )
        section("Tech support")
        support = fdf.groupby("TechSupport", observed=True)["churn_binary"].mean().sort_values() * 100
        st.dataframe(
            pd.DataFrame({"Churn rate": support.map(lambda x: f"{x:.1f}%")}),
            width="stretch",
        )
    with right:
        section("Payment method")
        payment = fdf.groupby("PaymentMethod", observed=True)["churn_binary"].mean().sort_values() * 100
        st.dataframe(
            pd.DataFrame({"Churn rate": payment.map(lambda x: f"{x:.1f}%")}),
            width="stretch",
        )
        section("Paperless billing")
        paperless = fdf.groupby("PaperlessBilling", observed=True)["churn_binary"].mean().sort_values() * 100
        st.dataframe(
            pd.DataFrame({"Churn rate": paperless.map(lambda x: f"{x:.1f}%")}),
            width="stretch",
        )

with data_tab:
    st.caption("First 250 rows of the filtered cohort.")
    st.dataframe(fdf.head(250), width="stretch", hide_index=True)

st.caption("All views above are descriptive associations. They do not establish causal effects.")
page_footer()
