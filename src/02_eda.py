"""Generate reproducible EDA figures and machine-readable findings."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import DATA_CLEAN, EDA_FINDINGS, FIGURES_DIR, TARGET

TEAL = "#0F766E"
AMBER = "#D97706"
RED = "#B91C1C"
BLUE = "#2563EB"
PURPLE = "#7C3AED"
GRID = "#E5E7EB"

plt.rcParams.update({
    "figure.dpi": 125,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby(column, observed=True)[TARGET].mean()


def style_axis(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def save_rate_bar(series: pd.Series, path, title: str, order=None) -> None:
    if order is not None:
        series = series.reindex(order)
    values = series * 100
    colors = [TEAL if v < 15 else AMBER if v < 30 else RED for v in values]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.bar(series.index.astype(str), values, color=colors)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title, weight="bold")
    ax.set_ylim(0, max(55, float(values.max()) + 8))
    style_axis(ax)
    for idx, value in enumerate(values):
        ax.text(idx, value + 1.0, f"{value:.1f}%", ha="center", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_CLEAN)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    findings: list[str] = []

    contract = churn_rate_by(df, "Contract")
    save_rate_bar(
        contract,
        FIGURES_DIR / "churn_by_contract.png",
        "Churn rate by contract type",
        ["Month-to-month", "One year", "Two year"],
    )
    findings.append(
        f"O1: Month-to-month churn is {contract['Month-to-month']*100:.1f}% vs "
        f"{contract['One year']*100:.1f}% (one-year) and {contract['Two year']*100:.1f}% (two-year)."
    )

    tenure_order = ["0-6", "7-12", "13-24", "25+"]
    tenure = churn_rate_by(df, "tenure_band")
    save_rate_bar(
        tenure,
        FIGURES_DIR / "churn_by_tenure_band.png",
        "Churn rate by tenure band",
        tenure_order,
    )
    findings.append(
        f"O2: Churn falls from {tenure['0-6']*100:.1f}% in months 0-6 to "
        f"{tenure['25+']*100:.1f}% after 25+ months."
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.hist(df.loc[df[TARGET] == 0, "MonthlyCharges"], bins=30, alpha=0.65,
            label="Retained", color=TEAL)
    ax.hist(df.loc[df[TARGET] == 1, "MonthlyCharges"], bins=30, alpha=0.65,
            label="Churned", color=RED)
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Customers")
    ax.set_title("Monthly charges: churned vs retained", weight="bold")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.savefig(FIGURES_DIR / "monthly_charges_by_churn.png")
    plt.close(fig)
    churn_bill = df.loc[df[TARGET] == 1, "MonthlyCharges"].mean()
    keep_bill = df.loc[df[TARGET] == 0, "MonthlyCharges"].mean()
    findings.append(
        f"O3: Churned customers average ${churn_bill:.2f}/month vs ${keep_bill:.2f}/month for retained customers."
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.hist(df["tenure"], bins=36, color=PURPLE, alpha=0.9)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    ax.set_title("Customer tenure distribution", weight="bold")
    style_axis(ax)
    fig.savefig(FIGURES_DIR / "tenure_distribution.png")
    plt.close(fig)
    findings.append(
        f"O4: Median tenure is {df['tenure'].median():.0f} months; "
        f"{(df['tenure'] <= 12).mean()*100:.1f}% of customers are within their first year."
    )

    corr_cols = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "num_services", "is_fiber", "is_month_to_month",
        "is_electronic_check", TARGET,
    ]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                square=False, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature correlations (analysis indicators included)", weight="bold")
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close(fig)
    top_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    findings.append(
        f"O5: Strongest linear associates with churn are {top_corr.index[0]} "
        f"(|r|={top_corr.iloc[0]:.2f}), {top_corr.index[1]} (|r|={top_corr.iloc[1]:.2f}), "
        f"and {top_corr.index[2]} (|r|={top_corr.iloc[2]:.2f})."
    )

    payment = churn_rate_by(df, "PaymentMethod")
    save_rate_bar(
        payment,
        FIGURES_DIR / "churn_by_payment_method.png",
        "Churn rate by payment method",
    )
    findings.append(
        f"O6: Electronic-check customers churn at {payment['Electronic check']*100:.1f}%."
    )

    tech = churn_rate_by(df, "TechSupport")
    save_rate_bar(
        tech,
        FIGURES_DIR / "churn_by_tech_support.png",
        "Churn rate by tech support status",
        ["Yes", "No", "No internet service"],
    )
    findings.append(
        f"O7: Customers without tech support churn at {tech['No']*100:.1f}% vs {tech['Yes']*100:.1f}% with tech support."
    )

    EDA_FINDINGS.write_text(
        "# EDA Findings\n\n" + "\n\n".join(f"- {item}" for item in findings) + "\n",
        encoding="utf-8",
    )
    for item in findings:
        print("-", item)
    print("eda OK | 7 figures written")


if __name__ == "__main__":
    main()
