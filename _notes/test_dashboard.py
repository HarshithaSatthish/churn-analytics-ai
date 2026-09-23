"""Smoke-test every dashboard page with streamlit AppTest (no browser needed)."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "app.py",
    "pages/1_Overview.py",
    "pages/2_Explore.py",
    "pages/3_Predictor.py",
    "pages/4_Model_and_Report.py",
]

failed = False
for page in PAGES:
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    at.run()
    if at.exception:
        failed = True
        print(f"FAIL {page}:")
        for exc in at.exception:
            print("   ", exc.stack_trace[-1] if exc.stack_trace else exc)
    else:
        print(f"OK   {page}")

raise SystemExit(1 if failed else 0)
