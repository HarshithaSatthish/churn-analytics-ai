"""Static deployment contract checks for Docker and Vercel packaging."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    required = [
        ROOT / "Dockerfile",
        ROOT / "Dockerfile.vercel",
        ROOT / "vercel.json",
        ROOT / "requirements.txt",
        ROOT / "requirements-runtime.txt",
        ROOT / ".python-version",
        ROOT / ".streamlit" / "config.toml",
        ROOT / "app.py",
        ROOT / "_shared.py",
        ROOT / "pages" / "1_Overview.py",
        ROOT / "pages" / "2_Explore.py",
        ROOT / "pages" / "3_Predictor.py",
        ROOT / "pages" / "4_Model_and_Report.py",
        ROOT / "data" / "clean" / "telco_clean.csv",
        ROOT / "models" / "churn_model.pkl",
        ROOT / "models" / "metrics.json",
    ]
    for path in required:
        check(path.exists(), f"Missing deployment asset: {path.relative_to(ROOT)}")

    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    check(config.get("fluid") is True, "Vercel Fluid compute must be enabled for Streamlit/WebSocket use.")

    docker_vercel = (ROOT / "Dockerfile.vercel").read_text(encoding="utf-8")
    check("streamlit run app.py" in docker_vercel, "Dockerfile.vercel does not start Streamlit.")
    check("${PORT:-80}" in docker_vercel, "Dockerfile.vercel must bind Streamlit to Vercel's PORT.")
    check("requirements-runtime.txt" in docker_vercel, "Dockerfile.vercel must use lean runtime requirements.")

    runtime_requirements = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
    for dependency in ("streamlit", "pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "joblib"):
        check(dependency in runtime_requirements, f"Missing runtime dependency: {dependency}")
    check("seaborn" not in runtime_requirements, "Training-only seaborn should not be in Vercel runtime requirements.")

    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    check(python_version.startswith("3.12"), "Deployment Python version should remain on 3.12.")

    forbidden = [ROOT / "_notes", ROOT / ".streamlit" / "secrets.toml"]
    for path in forbidden:
        check(not path.exists(), f"Forbidden submission/deployment artifact present: {path.relative_to(ROOT)}")

    print("DEPLOYMENT CHECK PASSED")
    print("Vercel: Dockerfile.vercel + dynamic PORT + Fluid compute")
    print("Runtime dependencies: lean set OK")


if __name__ == "__main__":
    main()
