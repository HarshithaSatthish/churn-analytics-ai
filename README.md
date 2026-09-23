# Customer Churn Analytics with AI — Final Project

**Author:** Akshay · **Date:** 2026-09-24
**Program:** IBM SkillsBuild Academic Internship (BharatCares / AICTE) — Data Analytics with AI

End-to-end churn analysis of 7,043 telecom customers: data cleaning, exploratory
analysis, a logistic-regression churn model (ROC-AUC 0.8448), an interactive
Streamlit dashboard with a live churn-risk predictor, and a structured business
report (Observations → Insights → Hypotheses → Recommendations).

AI assistance (Google Gemini) was used for code generation and analysis support,
per the masterclass workflow.

**Dataset:** IBM Telco Customer Churn (Apache-2.0)
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
(a copy is bundled at `data/raw/telco_churn.csv`)

## Run it (3 commands)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/01_clean.py && .venv/bin/python src/02_eda.py && .venv/bin/python src/03_model.py
.venv/bin/streamlit run app.py
```

Or open `notebooks/churn_project_colab.ipynb` in Google Colab — it runs the full
pipeline top-to-bottom with no manual steps.

## Project structure

```
data/raw/telco_churn.csv      # raw dataset (downloaded, never edited)
data/clean/telco_clean.csv    # cleaned dataset + data/clean/quality_log.md
src/01_clean.py               # cleaning + feature engineering
src/02_eda.py                 # 5 EDA figures -> figures/
src/03_model.py               # model -> models/churn_model.pkl + models/metrics.json
app.py + pages/               # 4-page Streamlit dashboard
reports/Final_Report.md        # structured report (their required format)
notebooks/                    # Colab-ready notebook
```

## Key results (test set, n=1409)

Accuracy 0.7395 · Precision 0.5059 · Recall 0.7995 · F1 0.6197 · ROC-AUC 0.8448

## Deliverables checklist

- [ ] IBM SkillsBuild learning-plan certificate submitted via form
- [ ] Masterclass attendance forms filled
- [ ] Final project (this package) submitted via project form
