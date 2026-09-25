"""Fresh-clone validation for data, artifacts, feature wiring, and inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd

from features import profile_to_model_row
from utils import (
    DATA_CLEAN,
    DATA_RAW,
    FEATURE_IMPORTANCE,
    FEATURE_INFO,
    METRICS_JSON,
    MODEL_PKL,
    MODEL_FEATURES,
    ROOT,
    TARGET,
)

REQUIRED_FILES = [
    DATA_RAW,
    DATA_CLEAN,
    MODEL_PKL,
    METRICS_JSON,
    FEATURE_INFO,
    FEATURE_IMPORTANCE,
    ROOT / "app.py",
    ROOT / "pages" / "1_Overview.py",
    ROOT / "pages" / "2_Explore.py",
    ROOT / "pages" / "3_Predictor.py",
    ROOT / "pages" / "4_Model_and_Report.py",
    ROOT / "README.md",
    ROOT / "reports" / "Final_Report.md",
    ROOT / ".streamlit" / "config.toml",
    ROOT / "Dockerfile.vercel",
    ROOT / "vercel.json",
    ROOT / "requirements-runtime.txt",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    for path in REQUIRED_FILES:
        check(path.exists(), f"Missing required file: {path.relative_to(ROOT)}")

    raw = pd.read_csv(DATA_RAW)
    clean = pd.read_csv(DATA_CLEAN)
    check(raw.shape == (7043, 21), f"Unexpected raw shape: {raw.shape}")
    check(len(clean) == 7043, f"Unexpected clean row count: {len(clean)}")
    check(clean["customerID"].is_unique, "customerID is not unique")
    check(clean[MODEL_FEATURES].isna().sum().sum() == 0, "NaNs in model features")
    check(set(clean[TARGET].unique()) == {0, 1}, "Unexpected target values")

    info = json.loads(FEATURE_INFO.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    check(info["model_features"] == MODEL_FEATURES, "feature_info schema drift")
    ranges = info.get("inference_ranges", {})
    check("MonthlyCharges" in ranges, "feature_info missing MonthlyCharges inference range")
    check(ranges["MonthlyCharges"]["min"] <= ranges["MonthlyCharges"]["max"],
          "Invalid MonthlyCharges inference range")
    check(metrics["threshold_selection"]["selected_on"] == "validation only",
          "Threshold must be validation-selected")

    model = joblib.load(MODEL_PKL)
    sample = clean[MODEL_FEATURES].iloc[[0]]
    score = float(model.predict_proba(sample)[0, 1])
    check(0 <= score <= 1, "Model score out of range")

    profile = {
        "gender": "Female", "senior": False, "partner": "Yes",
        "dependents": "No", "tenure": 12, "phone": "Yes", "multi": "No",
        "internet": "DSL", "onlinesec": "Yes", "onlinebak": "No",
        "devprot": "No", "techsup": "Yes", "tv": "No", "movies": "No",
        "contract": "One year", "paperless": "Yes",
        "paymethod": "Credit card (automatic)", "monthly": 55.0,
    }
    row = profile_to_model_row(profile)
    check(list(row.columns) == MODEL_FEATURES, "Predictor row schema mismatch")
    ui_score = float(model.predict_proba(row)[0, 1])
    check(0 <= ui_score <= 1, "Predictor score out of range")

    effects = pd.read_csv(FEATURE_IMPORTANCE)
    check(not effects.empty, "feature_importance.csv is empty")
    check({"feature", "coefficient", "odds_ratio", "comparison"}.issubset(effects.columns),
          "feature_importance.csv schema is incomplete")

    # Guard against documentation drifting away from the serialized final model.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "reports" / "Final_Report.md").read_text(encoding="utf-8")
    expected_tokens = [
        f"{metrics['roc_auc']:.4f}",
        f"{metrics['test_default']['accuracy']:.4f}",
        f"{metrics['test_default']['f1']:.4f}",
        f"{metrics['test_operating']['recall']:.4f}",
        f"{metrics['threshold_selection']['value']:.2f}",
    ]
    for token in expected_tokens:
        check(token in readme, f"README metric drift: missing {token}")
        check(token in report, f"Report metric drift: missing {token}")

    default_confusion = metrics["test_default"]
    check(
        all(str(default_confusion[key]) in readme for key in ("tn", "fp", "fn", "tp")),
        "README default confusion-matrix values drifted from metrics.json",
    )
    check(
        all(str(default_confusion[key]) in report for key in ("tn", "fp", "fn", "tp")),
        "Report default confusion-matrix values drifted from metrics.json",
    )

    print("VALIDATION PASSED")
    print(f"raw={raw.shape} clean={clean.shape} model_score={score:.4f} ui_score={ui_score:.4f}")
    print(
        f"ROC-AUC={metrics['roc_auc']:.4f} | cutoff="
        f"{metrics['threshold_selection']['value']:.2f} | "
        f"test recall={metrics['test_operating']['recall']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise
