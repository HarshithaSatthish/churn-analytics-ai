# OSS Recon — Data Analytics with AI final project (churn + Streamlit)
Researched 2026-09-23. Builder wants components to integrate directly, 1-day deadline.
Project shape: IBM Telco churn data → EDA → logistic regression (+ stronger model) → Streamlit dashboard → structured report.

## Direct-download dataset (verified, no auth)

**Primary:**
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

- Why trusted: appears verbatim in IBM's own Cloud Pak for Data code-pattern docs and multiple forks (mstand, mdurular forks of the IBM repo). IBM org repo, licensed Apache License 2.0.
- Note: an older tutorial variant without hyphens (`IBM/telcocustomerchurnonicp4d`) also circulates; the hyphenated one is confirmed in IBM's current docs. Prefer the hyphenated URL above.
- Sanity check after download: expect ~7,043 rows × 21 columns, ~977 KB. Spot-check that `TotalCharges` has 11 blank strings (tenure=0 customers) — if so, it's the genuine file.

**Fallback (dataset bundled in-repo, MIT-licensed repo):**
`https://raw.githubusercontent.com/farzadjenab/Customer-Churn-Prediction-in-Telecommunications/main/Telco-Customer-Churn.csv`
(repo page verified: default branch `main`, `Telco-Customer-Churn.csv` at repo root; raw URL is the standard GitHub pattern — builder should confirm it downloads.)

---

## (a) Streamlit dashboard templates

### 1. sruthik30/customer-churn-analytics
- URL: https://github.com/sruthik30/customer-churn-analytics
- License: **NONE DECLARED** (no LICENSE file in root — treat as all-rights-reserved; lift patterns, rewrite code)
- Stars/forks: 0 / 0 · Commits: 9 · Created 2026-07-01 · Updated ~Aug 2026 · Not archived
- Live demo: https://customer-churn-analytics-nfaikvg8ryrkwrnivqp8yb.streamlit.app (proves it runs)
- Structure: `train.py` (trains, saves joblib pipeline) + `app.py` + `notebooks/01_EDA.ipynb` + `model/churn_pipeline.pkl` + `style.css`
- Pages: Dashboard (KPI cards: churn rate, customer count, avg bill, avg tenure), Predict (customer form → churn risk + **SHAP explanation**), Analytics (churn by tenure/charges/internet/payment, correlation heatmap), Model (performance + feature importance), About
- What to lift: the multi-page layout skeleton, the SHAP-on-prediction-page pattern (big wow factor, low effort), the train.py → joblib artifact pattern
- Integration notes: rewrite in own code and restyle (no license). `streamlit-option-menu` dependency noted in its stack.
- Red flags: zero stars, dataset file is pre-cleaned (7,032 rows — builder should redo cleaning from the raw 7,043-row file to show the EDA work honestly); no license file.

### 2. harikris-1/ibm-telco-churn-prediction-dashboard
- URL: https://github.com/harikris-1/ibm-telco-churn-prediction-dashboard
- License: **NONE DECLARED** (no LICENSE in root)
- Stars/forks: 1 / 0 · Commits: 12 · Created 2026-07-10 · Updated ~72 days ago
- 5 tabs: EDA → SQL-based insights (SQLite via `load_db.py`) → statistical hypothesis tests (SciPy/statsmodels) → simulated A/B retention experiment → ML (Logistic Regression + Random Forest, ROC-AUC)
- What to lift: the stats-heavy tab structure — hypothesis tests + A/B simulation give the "insights/hypotheses" section of the required report real substance; the SQLite/SQL-insights angle is distinctive vs. other submissions
- Integration notes: needs user-supplied CSV (`data/telco_churn.csv`) — pair with the IBM raw URL above; `requirements.txt` pinned
- Red flags: no license; screenshot in README is a placeholder ("run the app and take a screenshot") — suggests less polish than claimed; user must wire data themselves.

### 3. zohaib-mzb/customer-churn-prediction-dashboard
- URL: https://github.com/zohaib-mzb/customer-churn-prediction-dashboard
- License: **unverified** (not checked — assume none declared)
- Updated ~29 days ago (freshest dashboard found); live demo: https://customerchurndashboardai.streamlit.app/
- XGBoost (scale_pos_weight) selected as production model (~73% F1, ~87% ROC-AUC) over LR/RF; SHAP explainability; Overview / Segment Analysis / Live Prediction pages; README ships a business findings table (segment → churn rate → insight)
- What to lift: the business-findings table format for the report's insights section; XGBoost + SHAP as the "AI" upgrade over plain logistic regression (matches the "Data Analytics with AI" theme)
- Red flags: license unverified; clone URL in its README points to a different repo name than the page slug — sloppy, verify before trusting structure.

---

## (b) EDA + modeling repos/notebooks

### 4. farzadjenab/customer-churn-prediction-in-telecommunications
- URL: https://github.com/farzadjenab/customer-churn-prediction-in-telecommunications
- License: **MIT** ✓ (badge + License section; repo page confirms License badge) — safest direct-adapt pick
- Stars/forks: 0 / 0 · Commits: 7 · Created 2026-06-28 · Updated ~83 days ago
- Single notebook `improved_churn_analysis.ipynb` + **dataset CSV bundled in-repo** + published research paper `MAIN.pdf`
- Methodology: TotalCharges fix, one-hot/label encoding, class_weight='balanced' (LR/RF) + scale_pos_weight (XGBoost), StandardScaler inside sklearn Pipeline for LR, 80/20 stratified split, 5-fold stratified CV, **McNemar's test** for pairwise model significance
- Results: LR ROC-AUC 0.8350 (recall 0.80), XGBoost F1 0.6211 — both strong; key drivers table (tenure ↓, fiber ↑, 2-year contract ↓, monthly charges ↑, electronic check ↑)
- What to lift: the whole preprocessing + model-comparison pipeline (MIT = fine to adapt with attribution); the key-drivers table and business recommendations section feed the report directly; the CV ± std tables give "rigor" credibility
- Red flags: zero stars despite quality; removes the 11 blank-TotalCharges rows rather than imputing 0 (minor — either defensible, just be consistent); no Streamlit piece (pair with #1/#2).

### 5. praiseemma/telco-customer-churn-prediction
- URL: https://github.com/praiseemma/telco-customer-churn-prediction
- License: **NONE DECLARED** (no LICENSE in root)
- Stars/forks: 0 / 0 · Commits: 16 · Created 2026-07-16 · Updated ~62 days ago · Live dashboard deployed
- 3 numbered notebooks (01_EDA → 02_logistic_regression (VIF, coefficients) → 03_random_forest (threshold tuning, SHAP)) + **`report/Telco_Churn_Consolidated_Report.docx` + `slides/Telco_Churn_Slide_Deck.pptx`** + `churn_model.pkl` + `scaler.pkl`
- Diagnose → Predict → Act framing; threshold tuned 0.5 → 0.25 for recall; concrete findings (month-to-month 42.7% vs 2.8% two-year; fiber+month-to-month 54.6%)
- What to lift: the **report structure** — the internship requires a structured written report, and this docx shows exactly what one looks like; the Diagnose/Predict/Act framing maps to observations→insights→hypotheses→recommendations
- Red flags: no license; pkl files are pre-trained artifacts — builder must retrain from scratch and document it (retraining from the bundled/IBM CSV is the honest path).

### 6. sohail413/customer-churn-prediction
- URL: https://github.com/sohail413/customer-churn-prediction
- License: **unverified** (not checked — assume none declared)
- Updated ~70 days ago; modular `src/` layout (`eda.py`, preprocessing pipeline, `encoders.pkl` persisted for inference-time reuse)
- Documents the classic EDA findings cleanly (TotalCharges blank→0.0, bimodal tenure, tenure↔TotalCharges 0.83 correlation) and applies **SMOTE on the training split only** (leakage-safe)
- What to lift: the preprocessing pipeline discipline (persist encoders, SMOTE train-only); the EDA findings list is a ready-made checklist for the notebook
- Red flags: license unverified; more code-engineering heavy — fine to borrow ideas, heavier to integrate whole.

### 7. hariomdubey01/ibm-telco-customer-churn-analytics
- URL: https://github.com/hariomdubey01/ibm-telco-customer-churn-analytics
- License: **unverified** (not checked)
- Updated **7 days ago** (freshest found); dataset CSV bundled (`data/Telco-Customer-Churn-Dataset.csv`)
- EDA notebook with chi-square statistical testing, customer segmentation, revenue analysis (not just charts — stats-backed insights); also ships Excel/Power BI artifacts (irrelevant here, but the insights carry over)
- What to lift: chi-square test snippets for the insights/hypotheses section — cheap rigor that most intern submissions skip
- Red flags: license unverified; "Future Enhancements" notes predictive modeling isn't done in-repo — use it for EDA/stats only.

---

## (c) EDA automation libraries (2026 status)

### 8. fg-data-profiling (formerly ydata-profiling, formerly pandas-profiling) — USE THIS
- `pip install fg-data-profiling` → `from data_profiling import ProfileReport`
- Status: **maintained**. Renamed April 2026; the old `ydata-profiling` package still installs but receives no updates (per Real Python, Sep 2026, and libraries.io migration notice). Supports Spark DataFrames too.
- One-line HTML report: distributions, correlations, missing-value maps, data-quality alerts — ideal for an EDA appendix/screenshots in the report.
- Red flags: heavy dependency tree (slow first install); full profile on 7k rows is fine but pass `minimal=True` if slow; watch pandas-2.x version pins in requirements.

### 9. Sweetviz — DO NOT USE
- Still mentioned in 2026 roundup articles, but **unmaintained since ~2022** (last release 2.2.1). Risky on modern pandas; no security/compat updates.

### Supporting libs (maintained, standard — pair with the above)
- **imbalanced-learn** (`imblearn`): SMOTE — apply to train split only. Maintained, standard.
- **shap**: model explainability for the dashboard prediction page. Maintained, standard.
- **plotly**: interactive charts in Streamlit. Maintained, standard.

---

## What to avoid (red-team notes)
1. **No-license repos (#1, #2, #5, #6, #7):** legally all-rights-reserved. Learn from them, rewrite — don't submit verbatim copies. Only #4 is MIT-clean.
2. **Stale IBM WML forks** (mstand, mdurular `telco-customer-churn-on-icp4d` forks, ~7 years old): Watson-ML-specific, outdated APIs — the raw CSV URL from IBM's repo is fine, the notebook code is not.
3. **Pre-trained .pkl artifacts** in repos (#1, #5): never submit someone else's pickle as your model. Retrain from the IBM CSV and save your own artifact.
4. **Sweetviz**: dead project, will break on modern pandas.
5. **Kaggle-only dataset instructions**: several READMEs tell you to download from Kaggle (login required) — use the IBM raw URL instead.
6. **Plagiarism risk**: the internship is graded partly on the report — the numbers/findings will look similar across submissions since it's the same dataset; differentiate with your own framing, your own dashboard UX, and at least one original angle (e.g., the A/B simulation from #2 or chi-square tests from #7).
