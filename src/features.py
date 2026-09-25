"""Pure feature-building helpers shared by training, tests, and the UI.

This module intentionally has no Streamlit dependency so feature wiring can be
unit-tested in a plain Python process.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

try:
    from .utils import MODEL_FEATURES
except ImportError:  # direct execution from src/
    from utils import MODEL_FEATURES

INTERNET_DEPENDENT_KEYS = (
    "onlinesec", "onlinebak", "devprot", "techsup", "tv", "movies"
)


def tenure_band(tenure: int) -> str:
    """Map tenure in months to the same bands used during data cleaning."""
    tenure = int(tenure)
    if tenure <= 6:
        return "0-6"
    if tenure <= 12:
        return "7-12"
    if tenure <= 24:
        return "13-24"
    return "25+"


def service_count(values: Iterable[str]) -> int:
    """Count service fields whose literal dataset value is 'Yes'."""
    return sum(1 for value in values if value == "Yes")


YES_NO = frozenset({"Yes", "No"})
YES_NO_NIS = frozenset({"Yes", "No", "No internet service"})
ALLOWED_CATEGORIES = {
    "gender": frozenset({"Female", "Male"}),
    "partner": YES_NO,
    "dependents": YES_NO,
    "phone": YES_NO,
    "multi": frozenset({"Yes", "No", "No phone service"}),
    "internet": frozenset({"DSL", "Fiber optic", "No"}),
    "onlinesec": YES_NO_NIS,
    "onlinebak": YES_NO_NIS,
    "devprot": YES_NO_NIS,
    "techsup": YES_NO_NIS,
    "tv": YES_NO_NIS,
    "movies": YES_NO_NIS,
    "contract": frozenset({"Month-to-month", "One year", "Two year"}),
    "paperless": YES_NO,
    "paymethod": frozenset({
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    }),
}


def validate_profile(profile: dict) -> list[str]:
    """Return human-readable errors for impossible or invalid profiles."""
    errors: list[str] = []
    tenure_value = int(profile["tenure"])
    monthly = float(profile["monthly"])

    if not 0 <= tenure_value <= 72:
        errors.append("Tenure must be between 0 and 72 months.")
    if monthly < 0:
        errors.append("Monthly charges cannot be negative.")

    if profile["phone"] == "No" and profile["multi"] != "No phone service":
        errors.append("Multiple lines must be 'No phone service' when phone service is No.")
    if profile["phone"] == "Yes" and profile["multi"] == "No phone service":
        errors.append("'No phone service' cannot be selected when phone service is Yes.")

    if profile["internet"] == "No":
        for key in INTERNET_DEPENDENT_KEYS:
            if profile[key] != "No internet service":
                errors.append(
                    "Internet add-ons must be 'No internet service' when internet service is No."
                )
                break
    else:
        for key in INTERNET_DEPENDENT_KEYS:
            if profile[key] == "No internet service":
                errors.append(
                    "Internet add-ons must be Yes/No when DSL or Fiber optic is selected."
                )
                break

    for field, allowed in ALLOWED_CATEGORIES.items():
        value = profile.get(field)
        if value not in allowed:
            errors.append(
                f"Unrecognized value {value!r} for {field}; "
                "expected one of: " + ", ".join(sorted(allowed)) + "."
            )
    return errors


def profile_to_model_row(profile: dict) -> pd.DataFrame:
    """Convert one UI profile into the exact model feature schema."""
    errors = validate_profile(profile)
    if errors:
        raise ValueError(" ".join(errors))

    tenure_value = int(profile["tenure"])
    yes_service_values = [
        profile["phone"], profile["multi"], profile["onlinesec"],
        profile["onlinebak"], profile["devprot"], profile["techsup"],
        profile["tv"], profile["movies"],
    ]

    row = {
        "gender": profile["gender"],
        "Partner": profile["partner"],
        "Dependents": profile["dependents"],
        "PhoneService": profile["phone"],
        "MultipleLines": profile["multi"],
        "InternetService": profile["internet"],
        "OnlineSecurity": profile["onlinesec"],
        "OnlineBackup": profile["onlinebak"],
        "DeviceProtection": profile["devprot"],
        "TechSupport": profile["techsup"],
        "StreamingTV": profile["tv"],
        "StreamingMovies": profile["movies"],
        "Contract": profile["contract"],
        "PaperlessBilling": profile["paperless"],
        "PaymentMethod": profile["paymethod"],
        "tenure_band": tenure_band(tenure_value),
        "SeniorCitizen": int(bool(profile["senior"])),
        "tenure": tenure_value,
        "MonthlyCharges": float(profile["monthly"]),
        "num_services": service_count(yes_service_values) + int(profile["internet"] != "No"),
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)

RAW_BATCH_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges",
]


def _as_bool(value) -> bool:
    """Accept 1/0, '1'/'0', or Yes/No spellings for SeniorCitizen."""
    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true"}: return True
    if text in {"0", "no", "n", "false", ""}: return False
    raise ValueError(f"Unrecognized SeniorCitizen value: {value!r}")


def raw_row_to_profile(row: pd.Series) -> dict:
    return {
        "gender": str(row["gender"]).strip(), "senior": _as_bool(row["SeniorCitizen"]),
        "partner": str(row["Partner"]).strip(), "dependents": str(row["Dependents"]).strip(),
        "tenure": int(float(row["tenure"])), "phone": str(row["PhoneService"]).strip(),
        "multi": str(row["MultipleLines"]).strip(), "internet": str(row["InternetService"]).strip(),
        "onlinesec": str(row["OnlineSecurity"]).strip(), "onlinebak": str(row["OnlineBackup"]).strip(),
        "devprot": str(row["DeviceProtection"]).strip(), "techsup": str(row["TechSupport"]).strip(),
        "tv": str(row["StreamingTV"]).strip(), "movies": str(row["StreamingMovies"]).strip(),
        "contract": str(row["Contract"]).strip(), "paperless": str(row["PaperlessBilling"]).strip(),
        "paymethod": str(row["PaymentMethod"]).strip(), "monthly": float(row["MonthlyCharges"]),
    }


def raw_frame_to_model_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list]:
    """Return model rows, row-level errors, and source index labels for valid rows."""
    missing = [c for c in RAW_BATCH_COLUMNS if c not in df.columns]
    if missing: raise ValueError("Upload is missing required columns: " + ", ".join(missing))
    frames, labels, errors = [], [], []
    for idx, row in df.iterrows():
        try:
            frames.append(profile_to_model_row(raw_row_to_profile(row)))
            labels.append(idx)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"row {idx}: {exc}")
    if not frames:
        raise ValueError("No valid customer rows found. " + (" ".join(errors[:3]) if errors else ""))
    return pd.concat(frames, ignore_index=True), errors, labels

