# Customer Churn Analytics with AI — Final Project

**Author:** Akshay  
**Date:** 2026-09-24  
**Program:** IBM SkillsBuild Academic Internship (BharatCares / AICTE) — Data Analytics with AI

A complete, reproducible customer-churn project built on the IBM Telco Customer Churn dataset. The repository includes data cleaning, exploratory analysis, an interpretable logistic-regression model, model evaluation with a leakage-safe train/validation/test workflow, a live Streamlit dashboard, a prediction/explanation workflow, a project report, and deployment files.

> Before final submission, verify that the author name and AI-assistance disclosure below match the actual submitter and your program's requirements.

## What this project delivers

- **7,043 customer records** cleaned with zero rows dropped.
- **Reproducible EDA** with generated charts and written findings.
- **Interpretable churn model** using a scikit-learn preprocessing + logistic-regression pipeline.
- **No threshold leakage:** the operating cutoff is selected on validation data only.
- **Untouched test evaluation:** ROC-AUC **0.8451**.
- **Operating cutoff:** **0.36**, selected to maximize validation precision while maintaining at least 70% validation recall.
- **Untouched-test operating metrics:** accuracy **0.7771**, precision **0.5649**, recall **0.6979**, F1 **0.6244**.
- **Interactive Streamlit UI:** overview, cohort explorer, logically constrained live predictor, explanation/what-if tools, and model/report page.
- **Deployment support:** Streamlit Community Cloud layout, Dockerfile, health check, CI workflow, and a fresh-clone validation script.

The bundled dataset is the IBM Telco Customer Churn dataset. A copy is stored at `data/raw/telco_churn.csv` so the project can be rebuilt without a runtime download.

## Repository structure

```text
churn-analytics-ai/
├── app.py                         # Streamlit landing page
├── _shared.py                     # shared Streamlit data/model/UI helpers
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Explore.py
│   ├── 3_Predictor.py
│   └── 4_Model_and_Report.py
├── src/
│   ├── 01_clean.py                # raw -> cleaned data
│   ├── 02_eda.py                  # reproducible EDA figures/findings
│   ├── 03_model.py                # train/validate/test + artifacts
│   ├── 04_validate.py             # fresh-clone integrity check
│   ├── features.py                # shared predictor/model feature contract
│   └── utils.py                   # paths and constants
├── data/
│   ├── raw/telco_churn.csv
│   └── clean/
├── models/
│   ├── churn_model.pkl
│   ├── metrics.json
│   ├── feature_info.json
│   └── feature_importance.csv
├── figures/                       # generated EDA/model figures
├── reports/Final_Report.md
├── notebooks/churn_project_colab.ipynb
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── Dockerfile
├── DEPLOYMENT.md
└── requirements.txt
```

## Run locally

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python src\01_clean.py
.\.venv\Scripts\python src\02_eda.py
.\.venv\Scripts\python src\03_model.py
.\.venv\Scripts\python src\04_validate.py
.\.venv\Scripts\streamlit run app.py
```

### macOS / Linux

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/01_clean.py
.venv/bin/python src/02_eda.py
.venv/bin/python src/03_model.py
.venv/bin/python src/04_validate.py
.venv/bin/streamlit run app.py
```

The validation step must print:

```text
VALIDATION PASSED
```

## Modeling methodology

The target is `churn_binary` (1 = churned, 0 = retained).

The data is split deterministically with `random_state=42`:

1. **20% untouched test set** is held out first.
2. The remaining 80% is split into **60% train** and **20% validation** overall.
3. A preprocessing + logistic-regression pipeline is fit on train data.
4. The operating threshold is chosen **only on validation data**.
5. The same pipeline specification is refit on train + validation.
6. The final serialized pipeline is evaluated once on the untouched test set.

Model inputs intentionally avoid exact duplicate indicator pairs such as `Contract` + `is_month_to_month` and `InternetService` + `is_fiber`. One-hot encoding drops a reference category, which makes the coefficient table more defensible. `TotalCharges` is cleaned for analysis but excluded from live prediction because it is largely determined by tenure and billing history and cannot be reliably reconstructed from a few form fields.

## Test-set results

| Metric | Standard cutoff 0.50 | Operating cutoff 0.36 |
|---|---:|---:|
| Accuracy | 0.8013 | 0.7771 |
| Precision | 0.6610 | 0.5649 |
| Recall | 0.5160 | 0.6979 |
| F1 | 0.5796 | 0.6244 |
| True positives | 193 | 261 |
| False positives | 99 | 201 |
| False negatives | 181 | 113 |
| True negatives | 936 | 834 |

Additional metrics: **ROC-AUC 0.8451**, average precision **0.6487**, Brier score **0.1366**. The majority-class accuracy baseline on the same test set is **0.7346**.

## Dashboard pages

- **Overview:** business KPIs and highest-churn segments.
- **Explore:** interactive cohort filters, segment views, billing/tenure distributions, service/payment comparisons, and CSV export.
- **Predictor:** live scoring with impossible service combinations prevented, a validation-selected retention flag, local model explanations, what-if scenarios, and an illustrative ROI calculator.
- **Model & Report:** untouched-test metrics, threshold-selection evidence, confusion matrix, ROC curve, interpretable coefficient effects, methodology, limitations, and downloadable report.

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md).

For Streamlit Community Cloud, deploy `app.py` from the repository root and use Python 3.12. The dependency file is `requirements.txt`, and `.streamlit/config.toml` is already included.

For Docker:

```bash
docker build -t churn-analytics-ai .
docker run --rm -p 8501:8501 churn-analytics-ai
```

## Reproducibility and integrity

Run this after cloning or before submission:

```bash
python src/01_clean.py
python src/02_eda.py
python src/03_model.py
python src/04_validate.py
```

`04_validate.py` checks required files, data shape, customer-ID uniqueness, model-feature nulls, feature-schema alignment, validation-only threshold selection, serialized-model inference, predictor-row wiring, and feature-effect artifact structure.

## Limitations

- This is a static public dataset, not a live telco production feed.
- There is no temporal holdout, so future-data drift is not measured.
- Dashboard explanations describe the fitted model, not causal effects.
- The predictor is suitable for project demonstration and prioritization logic, not autonomous customer treatment.
- Fairness, calibration drift, and production monitoring would need dedicated work before real-world use.

## AI-assistance disclosure

The original project states that Google Gemini was used for code-generation and analysis support following the masterclass workflow. Keep, edit, or expand this disclosure only if it accurately reflects the submitted work and your academic program's policy.
