"""Headless Streamlit smoke tests.

These run in CI where Streamlit is installed. They are skipped in minimal audit
sandboxes that only contain the scientific Python dependencies.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None

if STREAMLIT_AVAILABLE:
    from streamlit.testing.v1 import AppTest


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit is not installed in this environment")
class StreamlitSmokeTests(unittest.TestCase):
    def test_all_pages_render_without_exceptions(self):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        self.assertFalse(at.exception)

        for page in (
            "pages/1_Overview.py",
            "pages/2_Explore.py",
            "pages/3_Predictor.py",
            "pages/4_Model_and_Report.py",
        ):
            at.switch_page(page).run(timeout=15)
            self.assertFalse(at.exception, f"Streamlit page failed: {page}")

    def test_predictor_scores_default_profile(self):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        at.switch_page("pages/3_Predictor.py").run(timeout=15)
        self.assertFalse(at.exception)
        self.assertGreaterEqual(len(at.button), 1)
        at.button[0].click().run(timeout=15)
        self.assertFalse(at.exception)
        labels = [metric.label for metric in at.metric]
        self.assertIn("Churn risk score", labels)
        self.assertIn("Retention flag", labels)


if __name__ == "__main__":
    unittest.main()
