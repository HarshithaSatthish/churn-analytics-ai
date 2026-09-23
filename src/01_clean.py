"""Clean the bundled IBM Telco Customer Churn dataset.

Input:  data/raw/telco_churn.csv
Output: data/clean/telco_clean.csv and data/clean/quality_log.md
"""
from __future__ import annotations

import pandas as pd

from utils import (
    CATEGORICAL_COLS,
    DATA_CLEAN,
    DATA_RAW,
    NUMERIC_COLS,
    QUALITY_LOG,
    SERVICE_YES_COLS,
    TARGET,
)

EXPECTED_SHAPE = (7043, 21)
EXPECTED_RAW_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
}


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return an analysis-ready copy and number of blank TotalCharges fixed."""
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected raw shape {EXPECTED_SHAPE}, got {df.shape}.")
    if set(df.columns) != EXPECTED_RAW_COLUMNS:
        missing = sorted(EXPECTED_RAW_COLUMNS - set(df.columns))
        extra = sorted(set(df.columns) - EXPECTED_RAW_COLUMNS)
        raise ValueError(f"Unexpected raw schema. Missing={missing}; extra={extra}")
    if set(df["Churn"].dropna().unique()) != {"Yes", "No"}:
        raise ValueError("Churn must contain only Yes/No values.")
    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values found in raw data.")

    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    missing_total = out["TotalCharges"].isna()
    n_blank = int(missing_total.sum())
    if not (out.loc[missing_total, "tenure"] == 0).all():
        raise ValueError("Blank TotalCharges found for a customer with tenure > 0.")
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    out[TARGET] = (out["Churn"] == "Yes").astype("int8")
    out["tenure_band"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 200],
        labels=["0-6", "7-12", "13-24", "25+"],
    ).astype(str)
    out["num_services"] = (
        (out[SERVICE_YES_COLS] == "Yes").sum(axis=1)
        + out["InternetService"].ne("No").astype(int)
    ).astype("int8")

    # Analysis-only indicators. They are useful for correlations/EDA but are
    # deliberately excluded from the model to avoid duplicate information.
    out["is_fiber"] = (out["InternetService"] == "Fiber optic").astype("int8")
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype("int8")
    out["is_electronic_check"] = (out["PaymentMethod"] == "Electronic check").astype("int8")

    model_columns = CATEGORICAL_COLS + NUMERIC_COLS
    if len(out) != len(df):
        raise AssertionError("Cleaning changed the row count.")
    if int(out[model_columns].isna().sum().sum()) != 0:
        raise AssertionError("NaN values remain in model features.")
    return out, n_blank


def main() -> None:
    raw = pd.read_csv(DATA_RAW)
    clean, n_blank = clean_dataframe(raw)

    DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(DATA_CLEAN, index=False)

    churn_rate = float(clean[TARGET].mean())
    QUALITY_LOG.write_text(
        "# Data Quality Log\n\n"
        f"- Raw rows: {len(raw):,}; raw columns: {raw.shape[1]}\n"
        f"- Clean rows: {len(clean):,}; rows dropped: 0\n"
        f"- Duplicate customer IDs: {int(raw['customerID'].duplicated().sum())}\n"
        f"- Blank `TotalCharges` fixed: {n_blank} (all tenure = 0) -> 0.0\n"
        f"- Model-feature missing values after cleaning: 0\n"
        f"- Overall churn rate: {churn_rate:.4f} ({churn_rate * 100:.2f}%)\n"
        "- Engineered for analysis: `tenure_band`, `num_services`, `is_fiber`, "
        "`is_month_to_month`, `is_electronic_check`, `churn_binary`\n"
        "- Modeling note: exact duplicate indicator features are not used by the model.\n",
        encoding="utf-8",
    )
    print(
        f"clean OK | rows={len(clean)} | blank TotalCharges fixed={n_blank} | "
        f"churn={churn_rate:.4f}"
    )


if __name__ == "__main__":
    main()
