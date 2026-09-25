"""Live churn-risk scoring page with logically valid input states."""
import pandas as pd
import streamlit as st

from src.features import RAW_BATCH_COLUMNS, raw_frame_to_model_rows

from _shared import (
    RED,
    TEAL,
    cohort_scores,
    configure_page,
    contribution_bars,
    gauge,
    load_clean,
    load_feature_info,
    load_metrics,
    load_model,
    model_contributions,
    page_footer,
    profile_to_row,
    risk_band,
    section,
    sidebar_project_info,
)

configure_page("Predictor")
sidebar_project_info()
st.title("Churn predictor")
st.caption("Score one customer with the same feature contract used by the trained model.")

model = load_model()
metrics = load_metrics()
feature_info = load_feature_info()
cutoff = float(metrics["threshold_selection"]["value"])
monthly_range = feature_info.get("inference_ranges", {}).get(
    "MonthlyCharges", {"min": 0.0, "max": 500.0}
)
monthly_min = float(monthly_range["min"])
monthly_max = float(monthly_range["max"])

st.markdown(
    '<div class="callout"><b>Decision rule:</b> a score at or above '
    f'{cutoff:.0%} is flagged for retention outreach. The cutoff was selected on '
    'validation data before the untouched test evaluation.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Customer & account")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.checkbox("Senior citizen")
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    paymethod = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input(
        "Monthly charges ($)",
        min_value=monthly_min,
        max_value=monthly_max,
        value=min(max(70.0, monthly_min), monthly_max),
        step=1.0,
        help=f"Model-fit range: ${monthly_min:.2f} to ${monthly_max:.2f}.",
    )

with right:
    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    if phone == "No":
        multi = "No phone service"
        st.text_input("Multiple lines", value=multi, disabled=True)
    else:
        multi = st.selectbox("Multiple lines", ["No", "Yes"])

    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    if internet == "No":
        onlinesec = onlinebak = devprot = techsup = tv = movies = "No internet service"
        st.info("Internet add-ons are automatically set to 'No internet service'.")
    else:
        onlinesec = st.selectbox("Online security", ["No", "Yes"])
        onlinebak = st.selectbox("Online backup", ["No", "Yes"])
        devprot = st.selectbox("Device protection", ["No", "Yes"])
        techsup = st.selectbox("Tech support", ["No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No", "Yes"])
        movies = st.selectbox("Streaming movies", ["No", "Yes"])

profile = {
    "gender": gender,
    "senior": senior,
    "partner": partner,
    "dependents": dependents,
    "tenure": tenure,
    "phone": phone,
    "multi": multi,
    "internet": internet,
    "onlinesec": onlinesec,
    "onlinebak": onlinebak,
    "devprot": devprot,
    "techsup": techsup,
    "tv": tv,
    "movies": movies,
    "contract": contract,
    "paperless": paperless,
    "paymethod": paymethod,
    "monthly": monthly,
}

if st.button("Score churn risk", type="primary", width="stretch"):
    try:
        X = profile_to_row(profile)
        score = float(model.predict_proba(X)[0, 1])
        st.session_state["scored_customer"] = {"profile": dict(profile), "X": X, "score": score}
    except ValueError as exc:
        st.error(str(exc))

if "scored_customer" in st.session_state and st.session_state["scored_customer"]["profile"] != profile:
    st.session_state.pop("scored_customer", None)
    st.warning("Inputs changed since the last score. Click **Score churn risk** to refresh the result.")

if "scored_customer" in st.session_state:
    scored = st.session_state["scored_customer"]
    score = float(scored["score"])
    band, css, action = risk_band(score, cutoff)
    flagged = score >= cutoff

    section("Risk score")
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn risk score", f"{score:.1%}")
    m2.metric("Operating cutoff", f"{cutoff:.0%}")
    m3.metric("Retention flag", "YES" if flagged else "NO")
    gauge(score, cutoff)
    st.markdown(
        f'<div class="band {css}"><b>{band} RISK</b> — {action}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The score is a model estimate from a static public dataset. It is a prioritization aid, not a guarantee of churn."
    )

    output = scored["X"].copy()
    output["churn_risk_score"] = round(score, 4)
    output["operating_cutoff"] = cutoff
    output["retention_flag"] = int(flagged)
    output["risk_band"] = band
    st.download_button(
        "Download scored profile",
        output.to_csv(index=False).encode("utf-8"),
        file_name="scored_customer.csv",
        mime="text/csv",
    )

    explain_tab, whatif_tab, roi_tab = st.tabs(["Why this score", "What-if", "Portfolio ROI"])
    with explain_tab:
        st.markdown(
            "These are exact feature contributions to the logistic model's **log-odds** for this profile "
            "(the intercept is not displayed). They explain the model; they are not causal claims."
        )
        drivers, protectors = model_contributions(model, scored["X"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Pushing risk upward**")
            contribution_bars(drivers, RED)
        with col_b:
            st.markdown("**Protective contributions**")
            contribution_bars(protectors, TEAL)

    with whatif_tab:
        st.markdown("Change three scenario attributes and compare against the scored baseline.")
        w1, w2, w3 = st.columns(3)
        contract_options = ["Month-to-month", "One year", "Two year"]
        w_contract = w1.selectbox(
            "What-if contract",
            contract_options,
            index=contract_options.index(scored["profile"]["contract"]),
        )
        w_tenure = w2.slider("What-if tenure", 0, 72, int(scored["profile"]["tenure"]), key="whatif_tenure")
        w_monthly = w3.number_input(
            "What-if monthly charge ($)",
            min_value=monthly_min,
            max_value=monthly_max,
            value=float(scored["profile"]["monthly"]),
            step=1.0,
            key="whatif_monthly",
        )
        if st.button("Compare scenario"):
            changed = dict(scored["profile"])
            changed.update({"contract": w_contract, "tenure": w_tenure, "monthly": w_monthly})
            changed_score = float(model.predict_proba(profile_to_row(changed))[0, 1])
            delta = (changed_score - score) * 100
            c1, c2 = st.columns(2)
            c1.metric("Baseline", f"{score:.1%}")
            c2.metric("Scenario", f"{changed_score:.1%}", f"{delta:+.1f} pts")
            st.caption("Scenario differences are model responses, not estimated causal treatment effects.")

    with roi_tab:
        st.markdown(
            "Illustrative economics for contacting customers whose **current model score** exceeds the chosen campaign cutoff."
        )
        scores = cohort_scores()
        df = load_clean()
        campaign_cutoff = st.slider(
            "Campaign score cutoff", 0.10, 0.80, float(cutoff), 0.01, key="roi_cutoff"
        )
        selected = scores >= campaign_cutoff
        n_selected = int(selected.sum())
        avg_bill = float(df.loc[selected, "MonthlyCharges"].mean()) if n_selected else 0.0
        a, b, c = st.columns(3)
        contact_cost = a.number_input("Cost per contact ($)", 1.0, 100.0, 8.0, 1.0)
        save_rate = b.slider("Assumed save rate", 5, 50, 20) / 100
        annual_value = c.number_input(
            "Annual value per saved customer ($)",
            100.0,
            5000.0,
            max(100.0, round(avg_bill * 12, 2)),
            25.0,
        )
        saves = n_selected * save_rate
        campaign_cost = n_selected * contact_cost
        retained_value = saves * annual_value
        net_value = retained_value - campaign_cost
        r1, r2, r3 = st.columns(3)
        r1.metric("Customers contacted", f"{n_selected:,}")
        r2.metric("Expected saves", f"{saves:,.0f}")
        r3.metric("Illustrative net value", f"${net_value:,.0f}")
        st.caption("ROI is a scenario calculator based on user-entered assumptions; it is not a measured experiment result.")

section("Batch scoring")
st.caption(
    "Score many customers at once. Upload a CSV with the raw customer columns "
    "(same schema as the bundled dataset); each row is validated against the same "
    "feature contract as the single-customer form."
)
template = pd.DataFrame(columns=RAW_BATCH_COLUMNS)
st.download_button(
    "Download CSV template",
    template.to_csv(index=False).encode("utf-8"),
    file_name="batch_template.csv",
    mime="text/csv",
)
upload = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_upload")
if upload is not None:
    try:
        raw = pd.read_csv(upload)
        X_batch, batch_errors, valid_labels = raw_frame_to_model_rows(raw)
        batch_scores = model.predict_proba(X_batch)[:, 1]
        flagged = batch_scores >= cutoff
        bands = [risk_band(float(score_value), cutoff)[0] for score_value in batch_scores]
        valid_raw = raw.loc[valid_labels]

        b1, b2, b3 = st.columns(3)
        b1.metric("Customers scored", f"{len(X_batch):,}")
        b2.metric("Flagged for outreach", f"{int(flagged.sum()):,}")
        b3.metric("Mean churn score", f"{float(batch_scores.mean()):.1%}")
        if batch_errors:
            with st.expander(f"{len(batch_errors)} row(s) skipped"):
                st.write(batch_errors[:20])

        scored_out = pd.DataFrame(
            {
                "customerID": valid_raw["customerID"].tolist()
                if "customerID" in raw.columns
                else [f"row_{i}" for i in valid_labels],
                "churn_risk_score": [round(float(score_value), 4) for score_value in batch_scores],
                "risk_band": bands,
                "retention_flag": flagged.astype(int),
            }
        )
        st.dataframe(scored_out.head(100), width="stretch", hide_index=True)
        st.download_button(
            "Download scored customers",
            scored_out.to_csv(index=False).encode("utf-8"),
            file_name="scored_customers.csv",
            mime="text/csv",
            key="batch_download",
        )
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))


page_footer()
