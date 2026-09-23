# RED-TEAM AUDIT — Data Analytics with AI Project Plan v1.0
**Auditor:** independent red-team subagent · **Date:** 2026-09-23
**Plan under audit:** `~/workspace/your_files/Data_Analytics_AI_Project_Plan.md`
**Method:** read plan + OSS recon end-to-end; verified every checkable technical claim.
`pip --dry-run` dependency resolution of the full pin set: **PASS, zero conflicts.**
(PyPI confirms pandas 3.0.6, scikit-learn 1.9.1, seaborn 0.13.2 are current latest;
streamlit 1.63.0 and matplotlib 3.11.1 exist, one patch behind latest. A full runtime
smoke test was attempted but blocked — /tmp is a 512 MB tmpfs; root fs has 7.5 GB free,
so the builder must create the venv under the project dir or home, NOT /tmp.)

---

## CRITICAL

### C1. The SkillsBuild certificate is assumed done — never verified, zero minutes budgeted
The internship has THREE deliverables: (a) IBM SkillsBuild learning-plan certificate,
(b) masterclass attendance forms, (c) the project. The 12-hour plan budgets time only
for (c). Per the Masterclass 2 transcript, the certificate had to be submitted via form
by **August 31** — three weeks ago. If the boss hasn't completed the learning plan, that
is potentially several hours of coursework the plan doesn't account for; if the Aug 31
cutoff was hard, the internship certificate may already be unattainable and the entire
build is moot.
**Fix:** Boss confirms TONIGHT, before 08:00: (1) SkillsBuild learning plan completed?
(2) Certificate submitted? (3) Attendance forms done? If (1) is "no", the plan must be
rewritten around finishing the coursework first — the project is deliverable #3, not #1.

### C2. The deadline is an assumption, not a fact
"~1 day" comes from the boss's belief. No published deadline was found anywhere; the
Q&A session that was supposed to announce it never aired. The whole 08:00–20:00 schedule
is built on an unverified date.
**Fix:** Move the WhatsApp-group deadline check from 12:00 on build day to TONIGHT.
Do not start the build without a confirmed date + time + submission mechanism.

### C3. The submission-channel check is a boss dependency disguised as a builder task
§6 schedules "open the program WhatsApp group, locate the form" at 12:00–12:30 as if the
builder can do it. The builder (agent) has no access to that WhatsApp group — only the
boss does. If the boss is unreachable at noon, this step silently fails and "where do I
submit" gets discovered at 19:30, exactly what Risk #5 claims to prevent.
**Fix:** Explicitly assign to boss, deadline tonight. Builder proceeds on the
triple-package default (repo + zip + recording) regardless.

## MAJOR

### M1. Internal contradiction: confusion_matrix.png has no generator
§4 lists `figures/confusion_matrix.png`; §2 assigns ALL six PNGs to `src/02_eda.py`.
But a confusion matrix requires the trained model, which doesn't exist until
`src/03_model.py` runs. Followed literally, the stage contracts produce a report and
dashboard page referencing a file nothing generates → build break at QA.
**Fix:** Move confusion-matrix generation into `03_model.py` (it owns metrics.json too).
`02_eda.py` owns the other five.

### M2. Cross-dependency: the 12:00 EDA cut breaks the dashboard
§6 says if EDA overruns, cut `correlation_heatmap.png` first. But §8 Page 2 displays
"static correlation heatmap (from figures/)". Cut it and the dashboard references a
missing file.
**Fix:** Change the cut to `tenure_distribution.png` (nothing downstream references it),
or make the dashboard heatmap conditional. Never cut a file another stage consumes.

### M3. Cut order is backwards — the predictor page is the highest-wow artifact
Page 3 (Predictor: live churn probability) is cut FIRST at the 16:30 gate. But for a
project literally titled "Data Analytics with AI", the live predictor is the single most
distinctive, demo-able piece — it's what makes the submission feel like AI rather than
a static chart deck. Every other submission will have KPI cards and bar charts; few will
have a working predictor. Cutting it first surrenders the differentiation the OSS recon
explicitly recommended.
**Fix:** Cut Page 4's report expanders first (the .md report already exists standalone —
link it), keep the predictor. Second cut: Page 2's sidebar filters (static charts suffice).

### M4. No Colab notebook deliverable — the program taught Colab
The plan deliberately chooses local scripts over Colab (defensible: reproducibility).
But if the submission form asks for a Colab link or notebook file, scripts-only fails at
the final step and there is no time left to convert.
**Fix:** Add a 20-minute task (in the 17:00–18:30 block): export ONE `notebook.ipynb`
that runs the same steps top-to-bottom (or at minimum, verify tonight what the form
accepts). Do not leave this as an assumption.

### M5. OSS adaptation time is unbudgeted
"Before writing any module from scratch, check recon findings" — but zero minutes are
allocated to reading, understanding, and adapting the MIT notebook's pipeline. Honest
adaptation of someone else's code takes 30–60 min; the 08:30–10:00 and 12:30–14:00
blocks assume from-scratch velocity while depending on recon output.
**Fix:** Add a 30-min "recon review + attribution" block at 08:00, OR commit to writing
from scratch and use recon as reference only. Also add an attribution/credits line in
the README for the MIT-licensed adaptation (license compliance + honesty).

### M6. The 2.5-hour dashboard block is the tightest in the plan
Four pages (KPIs + donut + data dictionary / filters + 4 charts / predictor form /
metrics + coefficients + report expanders) in 150 minutes, for what is likely a
first-time Streamlit build. The hard stop exists, but the fallback isn't pre-designed —
decisions made under time pressure at 16:30 will be bad ones.
**Fix:** Pre-decide the 3-page fallback NOW: Overview / Explore / Predictor (per M3's
cut order). Write it into the plan so 16:30 is execution, not deliberation.

### M7. Dashboard double-builds every chart
`02_eda.py` renders matplotlib PNGs for the report; the dashboard re-implements the
same insights as native Streamlit charts. Two implementations of the same charts on a
1-day deadline.
**Fix:** Display the generated PNGs inside Streamlit via `st.image` — one charting
implementation, two consumers. Saves an estimated 30–45 min in the dashboard block.

## MINOR

### m1. `.gitignore` contradicts success criterion #8
`.gitignore` lists `*.mp4`, but "done" requires `demo_walkthrough.mp4` in the repo.
**Fix:** drop `*.mp4` from .gitignore, or store the video outside the repo (unlisted
YouTube link in README is the more robust delivery mechanism anyway — see m2).

### m2. Demo video as pass/fail is over-strict
If the recording fails, the project is still submittable. Demote criterion #8 to
should-have; the submission does not depend on it.

### m3. Plan vs. existing scaffold paths differ
The plan puts `app.py` + `pages/` at repo root; the already-scaffolded project has a
`dashboard/` directory. **Fix:** builder reconciles before 08:00 — pick one layout.

### m4. Move the import smoke test to 08:00
The plan's QA does the fresh-venv reinstall test at 18:30. If pandas-3.0/streamlit
runtime behavior breaks anything, discovering it at 18:30 is fatal. A 5-minute
`import pandas, sklearn, streamlit, seaborn` + tiny pipeline fit-test belongs in the
08:00 env block. (pip resolution verified clean; runtime behavior is the residual risk.)

### m5. RFM-as-proxy framing
Documenting "tenure ≈ recency" in Limitations is honest, but invoking RFM at all on a
dataset with no transaction history risks reading as padding to a sharp reviewer.
**Fix:** reframe as "behavioral segmentation features", keep the honesty note.

### m6. Report example numbers are borrowed
§9's "month-to-month ~42% vs ~3% two-year" examples come from another repo's findings.
They're labeled "e.g." and the plan correctly says real numbers come from metrics.json —
just ensure the builder doesn't copy them as findings.

### m7. Consider a cheap second model (tradeoff, not a demand)
The cut list bans Random Forest, but the recon's highest-value asset is its LR/RF/XGB
comparison. A 15-line, 20-minute RF comparison table inside the model block buys
"AI"-theme credibility cheaply while LR remains the shipped, interpretable model.
The pedagogical case for LR-only (what MC3 taught) is legitimate — this is a judgment
call, but it should be made deliberately, not by default.

---

## What the plan gets RIGHT (so the fixes stay in proportion)
- Falsifiable success criteria, hard time-boxes with named cuts, and a protected buffer.
- The 11-blank-TotalCharges fix (fill 0, don't drop) is correct and honest.
- Numbers-from-metrics.json discipline; no-rows-silently-dropped rule; random_state=42.
- Submission risk is named (Risk #5) — it just needs the boss-dependency made explicit (C3).
- Version pins verified: all five exist and resolve with zero conflicts (this audit).
