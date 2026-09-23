"""Model evaluation, explainability, methodology, and report page."""
from pathlib import Path

import pandas as pd
import streamlit as st

from _shared import (
    ROOT,
    configure_page,
    load_feature_effects,
    load_metrics,
    page_footer,
    readable_feature,
    section,
    sidebar_project_info,
)

configure_page("Model & Report")
sidebar_project_info()
st.title("Model & report")
st.caption("Evaluation is separated from model/threshold selection so the reported test result stays honest.")

metrics = load_metrics()
default = metrics["test_default"]
operating = metrics["test_operating"]
threshold = float(metrics["threshold_selection"]["value"])

section("Untouched test performance")
st.caption(
    f"Test rows: {metrics['split']['test_rows']:,} · churn rate: {metrics['split']['test_churn_rate']:.1%} · "
    f"majority-class accuracy baseline: {metrics['majority_baseline_accuracy']:.1%}"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
c2.metric("Accuracy @ 0.50", f"{default['accuracy']:.1%}")
c3.metric(f"Recall @ {threshold:.2f}", f"{operating['recall']:.1%}")
c4.metric(f"Precision @ {threshold:.2f}", f"{operating['precision']:.1%}")
c5.metric(f"F1 @ {threshold:.2f}", f"{operating['f1']:.3f}")

st.markdown(
    '<div class="callout"><b>Interpretation:</b> the model ranks churn risk well (ROC-AUC 0.845). '
    f'At the validation-selected {threshold:.0%} operating cutoff, it captures '
    f'{operating["tp"]} of {operating["tp"] + operating["fn"]} churners in the untouched test set, '
    f'with {operating["fp"]} false positives.</div>',
    unsafe_allow_html=True,
)

roc_path = ROOT / "figures" / "roc_curve.png"
cm_path = ROOT / "figures" / "confusion_matrix.png"
col1, col2 = st.columns(2)
with col1:
    if roc_path.exists():
        st.image(str(roc_path), caption="Untouched test ROC curve")
with col2:
    if cm_path.exists():
        st.image(str(cm_path), caption=f"Untouched test confusion matrix @ {threshold:.2f}")

section("How the operating cutoff was chosen")
st.markdown(
    f"The threshold was selected on the **validation split only**. The rule was: "
    f"maximize precision while maintaining at least 70% recall. That produced a cutoff of **{threshold:.2f}**. "
    "Only after freezing that value was the final pipeline evaluated on the test split."
)
threshold_path = ROOT / "figures" / "threshold_curve.png"
if threshold_path.exists():
    st.image(str(threshold_path), caption="Precision / recall / F1 on validation data only")

with st.expander("Default 0.50 vs operating cutoff"):
    compare = pd.DataFrame(
        [
            {"Cutoff": "0.50 (default)", **{k: default[k] for k in ["accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]}},
            {"Cutoff": f"{threshold:.2f} (operating)", **{k: operating[k] for k in ["accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]}},
        ]
    )
    st.dataframe(compare, use_container_width=True, hide_index=True)

section("Interpretable model effects")
st.markdown(
    "The pipeline uses logistic regression, one-hot encoding with a dropped reference category, and standardized numeric features. "
    "For categorical rows, the odds ratio compares with the listed reference. For numeric rows, it describes a +1 standard-deviation change."
)
effects = load_feature_effects().copy()
effects["Feature"] = effects["feature"].map(readable_feature)
effects["Direction"] = effects["coefficient"].map(lambda value: "Higher churn odds" if value > 0 else "Lower churn odds")
effects["Coefficient"] = effects["coefficient"].map(lambda value: f"{value:+.3f}")
effects["Odds ratio"] = effects["odds_ratio"].map(lambda value: f"{value:.2f}x")
effects = effects.rename(columns={"comparison": "Comparison"})
st.dataframe(
    effects[["Feature", "Direction", "Coefficient", "Odds ratio", "Comparison"]].head(15),
    use_container_width=True,
    hide_index=True,
)
st.caption("Coefficients are conditional model associations, not causal effects.")

section("Methodology")
st.markdown(
    "1. Clean the 7,043-row IBM Telco dataset and keep all rows.\n"
    "2. Hold out 20% as an untouched stratified test set.\n"
    "3. Split the remaining data into 60% train and 20% validation.\n"
    "4. Fit a preprocessing + logistic-regression pipeline on train.\n"
    "5. Select the retention cutoff on validation only.\n"
    "6. Refit the same pipeline specification on train + validation.\n"
    "7. Evaluate once on test and serialize that exact final pipeline."
)

section("Limitations")
st.markdown(
    "- Static public dataset with no time dimension; this is not temporal validation.\n"
    "- Customer risk can drift as prices, products, and behavior change.\n"
    "- The dashboard demonstrates prioritization; it does not prove retention interventions cause lower churn.\n"
    "- Fairness across protected or operationally sensitive groups requires a dedicated review before real-world use.\n"
    "- ROI values in the predictor are scenarios, not measured business outcomes."
)

section("Project report")
report_path = ROOT / "reports" / "Final_Report.md"
report_text = report_path.read_text(encoding="utf-8")
st.download_button(
    "Download Final_Report.md",
    data=report_text.encode("utf-8"),
    file_name="Final_Report.md",
    mime="text/markdown",
)
with st.expander("Read condensed report in the app"):
    st.markdown(report_text)

page_footer()
