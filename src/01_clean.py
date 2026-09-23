"""01_clean.py — raw IBM Telco CSV -> cleaned analysis-ready CSV.

Fixes:
  * TotalCharges arrives as text with 11 blanks (all tenure == 0, never billed)
    -> coerced to numeric, blanks filled with 0. Rows are NOT dropped.
Engineers: tenure_band, num_services, is_fiber, is_month_to_month,
is_electronic_check, churn_binary target.
Writes data/clean/telco_clean.csv + data/clean/quality_log.md
"""
import pandas as pd

from utils import (
    DATA_RAW, DATA_CLEAN, QUALITY_LOG, SERVICE_COLS, TARGET,
    CATEGORICAL_COLS, NUMERIC_COLS,
)


def main() -> None:
    df = pd.read_csv(DATA_RAW)
    raw_rows, raw_cols = df.shape

    # ---- acceptance checks on the raw file ----
    assert (raw_rows, raw_cols) == (7043, 21), f"unexpected shape {(raw_rows, raw_cols)}"
    assert set(df["Churn"].unique()) == {"Yes", "No"}, "unexpected Churn values"

    # ---- TotalCharges: text -> numeric, 11 blanks -> 0 ----
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = int(df["TotalCharges"].isna().sum())
    blank_tenures = df.loc[df["TotalCharges"].isna(), "tenure"]
    assert (blank_tenures == 0).all(), "blank TotalCharges found on tenure > 0"
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # ---- target ----
    df[TARGET] = (df["Churn"] == "Yes").astype(int)

    # ---- engineered features ----
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 200],
        labels=["0-6", "7-12", "13-24", "25+"],
    ).astype(str)
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype(int)
    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # ---- integrity assertions ----
    model_cols = CATEGORICAL_COLS + NUMERIC_COLS
    assert df.shape[0] == raw_rows, "rows were lost during cleaning"
    assert int(df[model_cols].isna().sum().sum()) == 0, "NaNs remain in model columns"

    DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_CLEAN, index=False)

    churn_rate = df[TARGET].mean()
    QUALITY_LOG.write_text(
        "# Data Quality Log\n\n"
        f"* Raw rows: {raw_rows}, raw cols: {raw_cols}\n"
        f"* Clean rows: {df.shape[0]} (0 rows dropped)\n"
        f"* Blank TotalCharges fixed: {n_blank} rows (all tenure == 0) -> filled with 0\n"
        f"* Overall churn rate: {churn_rate:.4f} ({churn_rate * 100:.2f}%)\n"
        f"* NaNs in model columns after cleaning: 0\n"
        "* Engineered: tenure_band, num_services, is_fiber, is_month_to_month, "
        "is_electronic_check, churn_binary\n"
    )
    print(f"clean OK: {df.shape[0]} rows, churn_rate={churn_rate:.4f}")


if __name__ == "__main__":
    main()
