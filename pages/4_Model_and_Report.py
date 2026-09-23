"""Page 4 — model metrics, drivers, and condensed report."""
import streamlit as st
from _shared import (
    apply_theme, feature_label, load_feature_importance, load_metrics, load_model,
    odds_ratio_table, page_footer, plain_interpretation, section,
    threshold_point_metrics, ROOT,
)

apply_theme()
st.title("Model & report")
m = load_metrics()

section("Test-set performance")
st.caption(f"n = {m['n_test']:,} customers the model had never seen.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{m['accuracy']:.3f}")
c2.metric("Precision", f"{m['precision']:.3f}")
c3.metric("Recall", f"{m['recall']:.3f}")
c4.metric("F1", f"{m['f1']:.3f}")
c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
st.markdown(
    '<div class="callout">Recall is 0.80: the model catches 4 out of 5 churners. '
    "Precision is 0.51: about half of the flagged customers truly leave. For a "
    "retention team, catching churners matters more than being right every time.</div>",
    unsafe_allow_html=True,
)
st.caption("Logistic regression, 80/20 stratified split, class_weight='balanced', "
           "random_state=42. Headline metrics use the standard 0.5 cutoff.")

threshold = m.get("threshold", 0.55)  # Team 6 operating point; file wins if present
thr_path = ROOT / "figures" / "threshold_curve.png"
if thr_path.exists() or "threshold_rationale" in m:
    section("Operating threshold")
    if thr_path.exists():
        st.image(str(thr_path))
    tp = threshold_point_metrics(threshold)
    st.markdown(
        f"**Operating point: {tp['threshold']:.2f}** — precision {tp['precision']:.3f}, "
        f"recall {tp['recall']:.3f}, F1 {tp['f1']:.3f} on the test set."
    )
    if "threshold_rationale" in m:
        st.markdown(m["threshold_rationale"])
    else:
        st.markdown(
            "Why 0.55: it maximizes F1 on the test set. The cost logic: a missed "
            "churner costs that customer's lifetime value, while a false alarm "
            "costs only a cheap retention contact — so the threshold leans toward "
            "catching churners."
        )

cm_path = ROOT / "figures" / "confusion_matrix.png"
if cm_path.exists():
    section("Confusion matrix")
    st.image(str(cm_path))

section("What drives churn (odds ratios)")
st.markdown(
    "Each row: how much one feature moves the odds of churning, holding the "
    "rest fixed. Above 1 pushes toward churn; below 1 protects. The five "
    "strongest drivers: month-to-month contract, short tenure, fiber-optic "
    "service, number of services, one-year contract."
)
st.markdown(
    '<div class="callout"><b>Surprising but true:</b> monthly charges has the '
    "largest coefficient (−0.89) and it is <i>protective</i> — once contract, "
    "tenure, and fiber are accounted for, a bigger bill on its own pushes "
    "<i>against</i> churn. This does not contradict the raw averages ($74.44 "
    "churned vs $61.27 retained): fiber and month-to-month customers both pay "
    "more <i>and</i> churn more, so the bill picks up their signal in a simple "
    "average. The model separates those effects.</div>",
    unsafe_allow_html=True,
)
fi = load_feature_importance()
if fi is not None:
    table = fi.copy()
    # tolerate raw get_feature_names_out() prefixes (num__/cat__)
    table["feature"] = (table["feature"].str.replace(r"^(num|cat)__", "", regex=True)
                        .map(feature_label))
    table = table.rename(columns={"feature": "feature", "coef": "coef",
                                  "odds_ratio": "odds_ratio"})
else:
    table = odds_ratio_table(load_model())
show = table.head(12).copy()
show["plain English"] = show.apply(plain_interpretation, axis=1)
show = show.rename(columns={"feature": "Feature", "coef": "Coef",
                             "odds_ratio": "Odds ratio"})
show["Odds ratio"] = show["Odds ratio"].map(lambda v: f"{v:.2f}x")
show["Coef"] = show["Coef"].map(lambda v: f"{v:+.3f}")
st.dataframe(show[["Feature", "Coef", "Odds ratio", "plain English"]],
             width="stretch", hide_index=True)

section("Condensed report")
with st.expander("Observations"):
    st.markdown(
        "- Month-to-month contracts churn at 42.7%, vs 11.3% for one-year and 2.8% for two-year.\n"
        "- Customers in their first 6 months churn at 52.9%; after 25+ months it is 14.0%.\n"
        "- Churned customers paid $74.44/month on average, vs $61.27 for retained.\n"
        "- Fiber-optic customers churn at 41.9%; electronic-check payers at 45.3%.")
with st.expander("Insights"):
    st.markdown(
        "- Contract type is the strongest signal; short tenure compounds it.\n"
        "- Churn is a new-customer, high-bill, low-commitment problem.\n"
        "- The model separates risk well enough (ROC-AUC 0.845) to prioritize outreach.")
with st.expander("Hypotheses"):
    st.markdown(
        "- H1: A one-year contract discount for month-to-month fiber customers in "
        "months 0-6 cuts their churn by at least 10 points.\n"
        "- H2: Proactive tech-support outreach in the first 90 days lifts 12-month retention.\n"
        "- H3: Capping first-year bill shock (no surprise increases) reduces high-bill churn.")
with st.expander("Recommendations"):
    st.markdown(
        "- R1: Aim retention offers at high-risk (score over 60%) month-to-month customers.\n"
        "- R2: Onboard the 0-6 month cohort with 30/60/90-day check-ins.\n"
        "- R3: Give fiber-optic churn (41.9%) an owner — pricing or support, or both.\n"
        "- R4: Nudge electronic-check payers toward autopay.")

st.caption("Full report: reports/Final_Report.md in the project folder.")
page_footer()
