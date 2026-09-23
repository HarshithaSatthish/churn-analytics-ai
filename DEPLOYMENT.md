# Deployment Guide

## Streamlit Community Cloud

This repository is organized for Streamlit Community Cloud:

- entrypoint: `app.py`
- Python dependencies: `requirements.txt` at repository root
- Streamlit configuration: `.streamlit/config.toml`
- all data/model/image paths are repository-relative
- no API keys or secrets are required

Deployment steps:

1. Push the project to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Select the target branch and set the entrypoint to `app.py`.
4. In Advanced settings, choose Python 3.12.
5. Deploy.

The application is intentionally Streamlit-only. There is no separate REST API or backend server to configure: the serialized scikit-learn pipeline is loaded in-process by Streamlit. Streamlit's built-in health route is used by the Docker health check.

## Docker

Build and run:

```bash
docker build -t churn-analytics-ai .
docker run --rm -p 8501:8501 churn-analytics-ai
```

Open `http://localhost:8501`.

## Fresh-clone verification

Before deployment or submission, run:

```bash
python src/01_clean.py
python src/02_eda.py
python src/03_model.py
python src/04_validate.py
```

The final command must print `VALIDATION PASSED`.
