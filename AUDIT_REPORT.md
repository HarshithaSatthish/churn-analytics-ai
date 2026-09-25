# Final Audit Report — Merged Submission Build

## Scope

This build was re-audited after GitHub access became available. The live `main` branch was inspected through commit `5bfc07bf81e16949c24cc860a1ecacd8d6470a3d`, then merged locally with the later Vercel/deployment hardening package. The live branch itself was not modified because the connected GitHub integration returned HTTP 403 for ref creation/write operations.

## GitHub findings retained

- `5bfc07b` fixes batch-scoring customer ID alignment when invalid upload rows are skipped.
- The same commit rejects categorical values outside the training vocabulary instead of allowing `OneHotEncoder(handle_unknown="ignore")` to silently map them to an all-zero category representation.
- The preceding hardening commit adds batch scoring, config-driven risk bands, flat headline metric keys, additional figure/report references, dataset-source documentation, and a faceless ≤3-minute demo walkthrough script.
- The report uses the required plain headings **Observations** and **Hypotheses**.

## Additional fixes in this merged build

- Preserves Vercel container deployment (`Dockerfile.vercel`, `vercel.json`, dynamic `$PORT`, Fluid compute).
- Preserves lean runtime requirements and Python 3.12 pin.
- Preserves deployment validator and Streamlit `AppTest` smoke tests.
- Preserves training-domain `MonthlyCharges` bounds in `feature_info.json` and uses them in single-customer and what-if inputs.
- Clears stale single-customer scores after form inputs change without blocking batch-scoring access.
- Keeps Overview statistics derived from the current dataset rather than hard-coded literals.
- Removes deprecated `use_container_width` calls in the dashboard pages.
- Adds the live GitHub batch-scoring regression tests to the expanded local test suite.

## Verification evidence

- Clean rebuild from raw CSV: PASS
- EDA regeneration: PASS
- Model retraining/serialization: PASS
- Leakage-safe threshold validation: PASS
- Deployment static contract: PASS
- 16 core tests: PASS
- 2 Streamlit AppTest tests: included for CI/runtime environment
- 5,000 randomized logically valid profiles: PASS

## Remaining human-owned submission items

The repository currently identifies the author as **Akshay**. The README also carries an AI-assistance disclosure. These are academic identity/policy declarations and must be verified by the actual submitter.

The repository now includes `reports/Demo_Walkthrough_Script.md`, but not a fabricated demo recording. Record the final deployed application and submit the actual MP4/link according to the program portal's rules.
