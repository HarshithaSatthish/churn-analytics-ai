# Demo walkthrough — recording script (≤ 3 minutes, faceless)

Record with any screen recorder (OBS, Xbox Game Bar, or phone screen-record of the
deployed Streamlit URL). No face needed — dashboard only, with voiceover.
Suggested total: ~2:45. Read the lines in *italics* while performing each shot.

## Setup (before pressing record)
1. Open the deployed Streamlit app URL in a maximized browser window.
2. Close other tabs. Zoom to 100%.
3. Have `batch_template.csv` (downloaded from the Predictor page) filled with 5
   sample rows ready on your desktop — or use the raw dataset's first rows.

## Shot list

**0:00–0:20 — Home**
- Show the landing page. Scroll slowly through the hero and project summary.
- *"This is my churn analytics project on 7,043 telecom customers. The goal: predict
  which customers will leave, and explain why — so retention teams know who to contact first."*

**0:20–0:50 — Overview**
- Click **Overview**. Point at the KPI cards (churn rate 26.5%, ROC-AUC 0.845).
- *"Overall churn is 26.5%. The model ranks at-risk customers with an ROC-AUC of
  0.845 on a test set it never saw during training."*

**0:50–1:20 — Explore**
- Click **Explore**. Change one filter (e.g. Contract = Month-to-month) and show the
  churn-rate chart updating.
- *"Month-to-month customers churn at 43%, versus 3% on two-year contracts.
  Early-tenure and fiber-optic customers show the same pattern."*

**1:20–2:00 — Predictor (single customer)**
- Click **Predictor**. Fill the form quickly: tenure 3, fiber optic, month-to-month,
  no tech support. Click **Score churn risk**.
- Point at the gauge, the risk band, and the "Why this score" contributions.
- *"A 3-month fiber customer scores about two-thirds risk and is flagged HIGH. The bars show
  which factors push the score up, straight from the model's coefficients."*
- Open the **What-if** tab, change contract to Two year, click Compare scenario.
- *"Changing the same profile to a two-year contract changes the model score. This is a model
  scenario comparison, not proof that the contract change itself causes retention."*

**2:00–2:30 — Batch scoring**
- Scroll to **Batch scoring**. Upload your prepared CSV, show the summary metrics
  and the scored table, click **Download scored customers**.
- *"For operations, a customer list can be validated, scored, banded, and flagged in one file."*

**2:30–2:45 — Model & Report (close)**
- Click **Model & Report**. Scroll past the ROC curve to the methodology.
- *"The cutoff was chosen on validation data only, then the model was evaluated once
  on an untouched test set — no leakage. The full report is downloadable from this page."*

## After recording
- Trim silence at the start/end. Export as MP4 and keep it under 3 minutes.
- Recommended filename: `demo_walkthrough.mp4`.
- If your submission portal expects the video separately, upload it there. If it explicitly
  requires the video inside the repository, remove the `*.mp4` ignore rule before committing it.
