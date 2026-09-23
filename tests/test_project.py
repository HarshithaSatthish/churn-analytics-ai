"""Small dependency-light tests for the project contract."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import joblib
import pandas as pd

from src.features import profile_to_model_row, validate_profile
from src.utils import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]


GOOD_PROFILE = {
    "gender": "Female", "senior": False, "partner": "Yes", "dependents": "No",
    "tenure": 12, "phone": "Yes", "multi": "No", "internet": "DSL",
    "onlinesec": "Yes", "onlinebak": "No", "devprot": "No", "techsup": "Yes",
    "tv": "No", "movies": "No", "contract": "One year", "paperless": "Yes",
    "paymethod": "Credit card (automatic)", "monthly": 55.0,
}


class ProjectContractTests(unittest.TestCase):
    def test_profile_schema_matches_model_contract(self):
        row = profile_to_model_row(GOOD_PROFILE)
        self.assertEqual(list(row.columns), MODEL_FEATURES)
        self.assertEqual(row.loc[0, "tenure_band"], "7-12")

    def test_impossible_internet_profile_is_rejected(self):
        profile = dict(GOOD_PROFILE)
        profile.update({"internet": "No", "onlinesec": "Yes"})
        errors = validate_profile(profile)
        self.assertTrue(any("Internet add-ons" in error for error in errors))

    def test_impossible_phone_profile_is_rejected(self):
        profile = dict(GOOD_PROFILE)
        profile.update({"phone": "No", "multi": "Yes"})
        errors = validate_profile(profile)
        self.assertTrue(any("Multiple lines" in error for error in errors))

    def test_internet_service_is_counted(self):
        row = profile_to_model_row(GOOD_PROFILE)
        # phone + active DSL + online security + tech support = 4 active services
        self.assertEqual(int(row.loc[0, "num_services"]), 4)

    def test_serialized_model_scores_ui_row(self):
        model = joblib.load(ROOT / "models" / "churn_model.pkl")
        score = float(model.predict_proba(profile_to_model_row(GOOD_PROFILE))[0, 1])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_threshold_was_not_selected_on_test(self):
        metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
        self.assertEqual(metrics["threshold_selection"]["selected_on"], "validation only")
        self.assertEqual(metrics["split"]["test_rows"], 1409)

    def test_clean_data_has_no_model_nulls(self):
        clean = pd.read_csv(ROOT / "data" / "clean" / "telco_clean.csv")
        self.assertEqual(int(clean[MODEL_FEATURES].isna().sum().sum()), 0)
        self.assertTrue(clean["customerID"].is_unique)

    def test_documented_metrics_match_final_artifact(self):
        metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
        readme = (ROOT / "README.md").read_text()
        report = (ROOT / "reports" / "Final_Report.md").read_text()
        tokens = [
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['test_default']['accuracy']:.4f}",
            f"{metrics['test_default']['f1']:.4f}",
            f"{metrics['test_operating']['recall']:.4f}",
        ]
        for token in tokens:
            self.assertIn(token, readme)
            self.assertIn(token, report)


if __name__ == "__main__":
    unittest.main()
