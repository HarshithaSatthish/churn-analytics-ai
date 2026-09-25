# Deployment Guide

The project has three supported deployment paths. **Vercel is now supported through its 2026 container runtime**; Streamlit remains available as a fallback.

## 1. Vercel — fast deployment target (2026 container/WebSocket path)

The repository includes:

- `Dockerfile.vercel` — runs Streamlit as a containerized HTTP server
- `vercel.json` — explicitly enables Fluid compute
- `requirements-runtime.txt` — lean runtime-only dependency set
- `.python-version` — pins Python 3.12 for non-container tooling

`Dockerfile.vercel` binds Streamlit to Vercel's injected `$PORT` rather than hard-coding port 8501. No separate API service is needed: Streamlit loads the serialized scikit-learn model in-process.

### Deploy from the Vercel dashboard

1. Push this project to GitHub.
2. In Vercel, choose **Add New → Project** and import the repository.
3. Keep the repository root as the project root.
4. Do not set an output directory or custom build command. Vercel detects the root `Dockerfile.vercel`.
5. Deploy.
6. After deployment, open the app and also verify `/_stcore/health` returns a healthy response.

After this updated code is pushed to the repository, you can also use: `https://vercel.com/new/clone?repository-url=https://github.com/HarshithaSatthish/churn-analytics-ai`

`vercel.json` sets `fluid: true` because Streamlit relies on a persistent WebSocket connection. Vercel's WebSocket and arbitrary-Docker HTTP-server support are recent 2026 platform features. WebSocket connections follow Vercel Function duration limits (for example, Hobby Fluid compute currently tops out at 5 minutes per invocation), so a long-lived Streamlit session can reconnect and in-memory session state can be lost if an instance is recycled. For a short project demo this path is practical; keep normal Docker/Streamlit Cloud as the stability fallback.

### Deploy with the Vercel CLI

```bash
npm i -g vercel
vercel
vercel --prod
```

No API keys or application secrets are required by this project. Vercel account authentication is the only deployment credential.

## 2. Streamlit Community Cloud

This repository remains organized for Streamlit Community Cloud:

- entrypoint: `app.py`
- Python dependencies: `requirements.txt` at repository root
- Streamlit configuration: `.streamlit/config.toml`
- all data/model/image paths are repository-relative
- no application secrets are required

Deployment steps:

1. Push the project to GitHub.
2. Create a new Streamlit app from the repository.
3. Select the target branch and set the entrypoint to `app.py`.
4. Choose Python 3.12 if the deployment UI exposes the Python-version option.
5. Deploy.

## 3. Docker

Build and run locally or on any container host:

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
python src/05_deployment_check.py
python -m unittest discover -s tests -v
```

The critical outputs are:

```text
VALIDATION PASSED
DEPLOYMENT CHECK PASSED
```
