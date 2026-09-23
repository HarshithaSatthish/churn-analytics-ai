"""02_eda.py — clean CSV -> 5 PNG figures + figures/eda_findings.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import DATA_CLEAN, FIGURES_DIR, EDA_FINDINGS, TARGET

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(col, observed=True)[TARGET].mean().sort_values()


def main() -> None:
    df = pd.read_csv(DATA_CLEAN)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    findings = []

    # 1. churn rate by contract
    s = churn_rate_by(df, "Contract")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s.index, s.values * 100, color=["#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by contract type")
    for i, v in enumerate(s.values * 100):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10)
    fig.savefig(FIGURES_DIR / "churn_by_contract.png")
    plt.close(fig)
    findings.append(
        f"O1: Month-to-month contracts churn at {s['Month-to-month']*100:.1f}%, "
        f"vs {s['One year']*100:.1f}% (one-year) and {s['Two year']*100:.1f}% (two-year)."
    )

    # 2. churn rate by tenure band
    order = ["0-6", "7-12", "13-24", "25+"]
    s = df.groupby("tenure_band", observed=True)[TARGET].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s.index, s.values * 100, color="#1f77b4")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by tenure band (months)")
    for i, v in enumerate(s.values * 100):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10)
    fig.savefig(FIGURES_DIR / "churn_by_tenure_band.png")
    plt.close(fig)
    findings.append(
        f"O2: Customers in their first 6 months churn at {s['0-6']*100:.1f}%, "
        f"falling to {s['25+']*100:.1f}% after 25+ months — churn is a new-customer problem."
    )

    # 3. monthly charges distribution split by churn
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.loc[df[TARGET] == 0, "MonthlyCharges"], bins=30, alpha=0.6, label="Retained")
    ax.hist(df.loc[df[TARGET] == 1, "MonthlyCharges"], bins=30, alpha=0.6, label="Churned")
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Customers")
    ax.set_title("Monthly charges distribution: churned vs retained")
    ax.legend()
    fig.savefig(FIGURES_DIR / "monthly_charges_by_churn.png")
    plt.close(fig)
    m_churn = df.loc[df[TARGET] == 1, "MonthlyCharges"].mean()
    m_keep = df.loc[df[TARGET] == 0, "MonthlyCharges"].mean()
    findings.append(
        f"O3: Churned customers paid ${m_churn:.2f}/mo on average vs ${m_keep:.2f}/mo "
        "for retained — higher bills correlate with churn."
    )

    # 4. tenure distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["tenure"], bins=36, color="#9467bd")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    ax.set_title("Customer tenure distribution")
    fig.savefig(FIGURES_DIR / "tenure_distribution.png")
    plt.close(fig)
    findings.append(
        f"O4: Median tenure is {df['tenure'].median():.0f} months; "
        f"{(df['tenure'] <= 12).mean()*100:.1f}% of customers are within their first year."
    )

    # 5. correlation heatmap (numeric features + target)
    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
                "num_services", "is_fiber", "is_month_to_month",
                "is_electronic_check", TARGET]
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlations (incl. churn)")
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close(fig)
    top_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    findings.append(
        f"O5: Strongest linear associates of churn: {top_corr.index[0]} "
        f"(|r|={top_corr.iloc[0]:.2f}), {top_corr.index[1]} (|r|={top_corr.iloc[1]:.2f}), "
        f"{top_corr.index[2]} (|r|={top_corr.iloc[2]:.2f})."
    )

    # fiber check (feeds the report's headline insight)
    fiber_churn = df.loc[df["is_fiber"] == 1, TARGET].mean()
    findings.append(
        f"O6: Fiber-optic customers churn at {fiber_churn*100:.1f}%."
    )

    EDA_FINDINGS.write_text(
        "# EDA Findings\n\n" + "\n\n".join(f"- {f}" for f in findings) + "\n"
    )
    for f_ in findings:
        print("-", f_)
    print("eda OK: 5 figures written")


if __name__ == "__main__":
    main()
