"""Page 3 — live churn-risk predictor."""
import pandas as pd
import streamlit as st
from _shared import (
    apply_theme, load_clean, load_feature_info, load_model, cohort_scores,
    contribution_breakdown, contribution_bars, gauge, page_footer,
    profile_to_row, risk_band, section, RED, TEAL,
)

apply_theme()
st.title("Churn predictor")
st.markdown("Enter a customer profile. The trained model scores the churn risk live.")

info = load_feature_info()
model = load_model()

with st.form("predict"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.checkbox("Senior citizen")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    with c2:
        onlinesec = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        onlinebak = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        devprot = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        techsup = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        paymethod = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", 0.0, 500.0, 70.0, 1.0)
    submitted = st.form_submit_button("Score churn risk")

if submitted:
    profile = {
        "gender": gender, "senior": senior, "partner": partner,
        "dependents": dependents, "tenure": tenure, "phone": phone,
        "multi": multi, "internet": internet, "onlinesec": onlinesec,
        "onlinebak": onlinebak, "devprot": devprot, "techsup": techsup,
        "tv": tv, "movies": movies, "contract": contract,
        "paperless": paperless, "paymethod": paymethod, "monthly": monthly,
    }
    X = profile_to_row(profile)
    proba = float(model.predict_proba(X)[0, 1])
    st.session_state["scored"] = {"profile": profile, "X": X, "proba": proba}

if st.session_state.get("scored"):
    s = st.session_state["scored"]
    proba = s["proba"]
    band, css, action = risk_band(proba)

    section("Risk score")
    st.markdown(f"### {proba * 100:.1f}% churn probability")
    gauge(proba)
    st.markdown(
        f'<div class="band {css}"><b>Risk: {band}</b> — {action}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Risk bands: under 30% = low, 30-60% = medium, over 60% = high.")

    out = s["X"].copy()
    out["churn_probability"] = round(proba, 4)
    out["risk_band"] = band
    st.download_button(
        "Download this scored profile (CSV)",
        out.to_csv(index=False).encode(),
        file_name="churn_scored_profile.csv",
        mime="text/csv",
    )

    with st.expander("Why this score"):
        st.markdown(
            "Top factors pushing this customer toward churn, and what protects "
            "them. Computed exactly from the model's coefficients — no black box."
        )
        drivers, protectors = contribution_breakdown(model, s["X"])
        st.markdown("**Pushing toward churn**")
        contribution_bars(drivers, RED)
        st.markdown("**Protective factors**")
        contribution_bars(protectors, TEAL)

    with st.expander("What-if: try a retention offer"):
        st.markdown(
            "Change one or two things and see how the score moves. "
            "Example: what if this customer moved to a one-year contract?"
        )
        w1, w2, w3 = st.columns(3)
        w_contract = w1.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                 index=["Month-to-month", "One year", "Two year"].index(
                                     s["profile"]["contract"]))
        w_tenure = w2.slider("Tenure (months)", 0, 72, int(s["profile"]["tenure"]),
                             key="w_tenure")
        w_monthly = w3.number_input("Monthly charges ($)", 0.0, 500.0,
                                    float(s["profile"]["monthly"]), 1.0, key="w_monthly")
        if st.button("Compare with baseline"):
            wp = dict(s["profile"])
            wp.update({"contract": w_contract, "tenure": w_tenure, "monthly": w_monthly})
            w_proba = float(model.predict_proba(profile_to_row(wp))[0, 1])
            delta = (w_proba - proba) * 100
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Baseline:** {proba * 100:.1f}% → "
                        f"**What-if:** {w_proba * 100:.1f}%")
            if delta < 0:
                st.success(f"Risk drops by {abs(delta):.1f} points with this change.")
            elif delta > 0:
                st.error(f"Risk rises by {delta:.1f} points with this change.")
            else:
                st.info("No change in risk.")
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Retention ROI calculator"):
        st.markdown(
            "Apply the model to the full customer base. How much could a "
            "retention campaign on high-risk customers be worth?"
        )
        scores = cohort_scores()
        df = load_clean()
        high = scores >= 0.60
        n_high = int(high.sum())
        avg_bill = float(df.loc[high, "MonthlyCharges"].mean()) if n_high else 0.0
        st.caption(f"High-risk customers in the data (score over 60%): {n_high:,}.")
        r1, r2, r3 = st.columns(3)
        cost = r1.slider("Cost per contacted customer ($)", 1, 50, 8)
        save_rate = r2.slider("Save rate — % of contacted who stay", 5, 50, 25)
        annual_value = r3.number_input("Annual value per saved customer ($)",
                                       100.0, 5000.0, round(avg_bill * 12, 2), 10.0)
        saves = n_high * save_rate / 100
        gross = saves * annual_value
        spend = n_high * cost
        net = gross - spend
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f"**Expected saves:** {saves:,.0f} customers<br>"
            f"**Value of saves:** ${gross:,.0f}<br>"
            f"**Campaign cost:** ${spend:,.0f}<br>"
            f"**Expected net:** ${net:,.0f}",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if net > 0:
            st.success("The campaign pays for itself under these assumptions.")
        else:
            st.warning("Under these assumptions the campaign costs more than it saves.")

st.caption("Demo on the trained logistic-regression model — not a production system.")
page_footer()
