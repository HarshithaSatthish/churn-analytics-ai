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
