"""Train, validate, and serialize the churn model without test-set leakage.

Methodology
-----------
1. Hold out 20% of the data as an untouched stratified test set.
2. Split the remaining 80% into train (60% total) and validation (20% total).
3. Fit an interpretable logistic regression on the train split.
4. Choose an operating threshold on validation only: highest precision while
   maintaining at least 70% recall.
5. Refit the same model specification on train+validation data.
6. Evaluate once on the untouched test set and save that exact fitted pipeline.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    CATEGORICAL_COLS,
    DATA_CLEAN,
    FEATURE_IMPORTANCE,
    FEATURE_INFO,
    FIGURES_DIR,
    METRICS_JSON,
    MIN_TARGET_RECALL,
    MODEL_DIR,
    MODEL_FEATURES,
    MODEL_PKL,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
)


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
                CATEGORICAL_COLS,
            ),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def metric_block(y_true, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def choose_threshold(y_val, proba_val: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose a validation-only cutoff for a retention use case."""
    rows: list[dict] = []
    for threshold in np.round(np.arange(0.10, 0.61, 0.01), 2):
        block = metric_block(y_val, proba_val, float(threshold))
        rows.append(block)
    sweep = pd.DataFrame(rows)

    eligible = sweep[sweep["recall"] >= MIN_TARGET_RECALL]
    if eligible.empty:
        chosen = sweep.sort_values(["recall", "precision"], ascending=False).iloc[0]
    else:
        chosen = eligible.sort_values(
            ["precision", "f1", "threshold"], ascending=False
        ).iloc[0]
    return float(chosen["threshold"]), sweep


def feature_effects(pipe: Pipeline) -> pd.DataFrame:
    """Create an interpretable coefficient table for the serialized model."""
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]
    coefficients = clf.coef_[0]

    encoder: OneHotEncoder = pre.named_transformers_["cat"]
    scaler: StandardScaler = pre.named_transformers_["num"]

    rows: list[dict] = []
    coef_index = 0

    for col, categories, drop_idx in zip(
        CATEGORICAL_COLS, encoder.categories_, encoder.drop_idx_
    ):
        dropped = categories[int(drop_idx)] if drop_idx is not None else None
        for idx, category in enumerate(categories):
            if drop_idx is not None and idx == int(drop_idx):
                continue
            coef = float(coefficients[coef_index])
            rows.append(
                {
                    "feature": f"{col}={category}",
                    "source_feature": col,
                    "coefficient": round(coef, 6),
                    "odds_ratio": round(float(np.exp(coef)), 6),
                    "effect_type": "categorical",
                    "comparison": f"vs {dropped}",
                }
            )
            coef_index += 1

    for col, scale in zip(NUMERIC_COLS, scaler.scale_):
        coef = float(coefficients[coef_index])
        rows.append(
            {
                "feature": col,
                "source_feature": col,
                "coefficient": round(coef, 6),
                "odds_ratio": round(float(np.exp(coef)), 6),
                "effect_type": "numeric_standardized",
                "comparison": f"per +1 SD ({float(scale):.2f} original units)",
            }
        )
        coef_index += 1

    effects = pd.DataFrame(rows)
    effects["abs_coefficient"] = effects["coefficient"].abs()
    return effects.sort_values("abs_coefficient", ascending=False).drop(
        columns="abs_coefficient"
    )


def main() -> None:
    df = pd.read_csv(DATA_CLEAN)
    X = df[MODEL_FEATURES]
    y = df[TARGET]

    # 80/20 outer split: the test set is never used for threshold selection.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # 75/25 split of the remaining 80% => 60/20/20 overall.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    selector_model = build_pipeline()
    selector_model.fit(X_train, y_train)
    val_proba = selector_model.predict_proba(X_val)[:, 1]
    threshold, sweep = choose_threshold(y_val, val_proba)
    validation_metrics = metric_block(y_val, val_proba, threshold)

    # Refit on all development data after the threshold is frozen.
    final_model = build_pipeline()
    final_model.fit(X_train_val, y_train_val)
    test_proba = final_model.predict_proba(X_test)[:, 1]

    default_metrics = metric_block(y_test, test_proba, 0.50)
    operating_metrics = metric_block(y_test, test_proba, threshold)
    majority_accuracy = float(max(y_test.mean(), 1 - y_test.mean()))

    metrics = {
        "model": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "split": {
            "strategy": "stratified 60/20/20 train/validation/test",
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "test_churn_rate": round(float(y_test.mean()), 4),
        },
        "threshold_selection": {
            "selected_on": "validation only",
            "objective": (
                f"maximize precision subject to recall >= {MIN_TARGET_RECALL:.0%}"
            ),
            "value": round(float(threshold), 2),
            "validation_metrics": validation_metrics,
        },
        "test_default": default_metrics,
        "test_operating": operating_metrics,
        "roc_auc": round(float(roc_auc_score(y_test, test_proba)), 4),
        "average_precision": round(float(average_precision_score(y_test, test_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_test, test_proba)), 4),
        "majority_baseline_accuracy": round(majority_accuracy, 4),
        "feature_count_raw": len(MODEL_FEATURES),
        "notes": (
            "Default 0.50 metrics describe standard classification. The operating cutoff "
            "was selected on validation data before the final untouched test evaluation."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PKL)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    FEATURE_INFO.write_text(
        json.dumps(
            {
                "categorical": CATEGORICAL_COLS,
                "numeric": NUMERIC_COLS,
                "model_features": MODEL_FEATURES,
                "target": TARGET,
                "operating_threshold": round(float(threshold), 2),
                "one_hot_drop": "first",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    feature_effects(final_model).to_csv(FEATURE_IMPORTANCE, index=False)

    # Operating confusion matrix (the action-oriented view used in the app).
    operating_pred = (test_proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, operating_pred),
        display_labels=["Retained", "Churned"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Test confusion matrix @ threshold {threshold:.2f}", weight="bold")
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # ROC curve on the untouched test set.
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot(fpr, tpr, linewidth=2.2, label=f"Model (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — untouched test set", weight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(FIGURES_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close(fig)

    # Threshold curve is validation-only to make the selection process explicit.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision")
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall")
    ax.plot(sweep["threshold"], sweep["f1"], label="F1")
    ax.axhline(MIN_TARGET_RECALL, linestyle=":", linewidth=1.2,
               label=f"Recall target ({MIN_TARGET_RECALL:.0%})")
    ax.axvline(threshold, linestyle="--", linewidth=1.5,
               label=f"Chosen cutoff ({threshold:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Operating-threshold selection — validation set only", weight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.savefig(FIGURES_DIR / "threshold_curve.png", bbox_inches="tight")
    plt.close(fig)

    # Serialization smoke test.
    loaded = joblib.load(MODEL_PKL)
    sample_proba = float(loaded.predict_proba(X_test.iloc[[0]])[0, 1])
    if not 0.0 <= sample_proba <= 1.0:
        raise AssertionError("Serialized model returned an invalid probability.")

    print(json.dumps(metrics, indent=2))
    print(f"model OK | operating threshold={threshold:.2f} | sample={sample_proba:.4f}")


if __name__ == "__main__":
    main()
