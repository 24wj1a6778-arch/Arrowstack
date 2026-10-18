"""Exploratory data analysis on the TRAIN split only (avoids peeking at test).

Run:  python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data_prep import load_split
from src.features import add_features


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    df = add_features(train.drop(columns=[config.TARGET]))
    df[config.TARGET] = train[config.TARGET].values

    overall = df[config.TARGET].mean()
    lines = ["# EDA summary (train split)", "",
             f"- Rows: {len(df):,} | Overall churn rate: **{overall:.1%}** (class imbalance -> use PR-AUC, not accuracy)",
             f"- Missing values: {df.isna().sum()[df.isna().sum() > 0].to_dict() or 'none'}", ""]

    cols = ["Contract", "InternetService", "PaymentMethod", "tenure_group", "TechSupport", "SeniorCitizen"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cols):
        rate = df.groupby(col)[config.TARGET].mean().sort_values()
        rate.plot.barh(ax=ax, color="#2b6cb0")
        ax.axvline(overall, color="red", ls="--", lw=1)
        ax.set_title(f"Churn rate by {col}"); ax.set_xlabel("churn rate"); ax.set_ylabel("")
        lines.append(f"- Churn by **{col}**: " + ", ".join(f"{k}={v:.0%}" for k, v in rate.items()))
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "eda_churn_by_segment.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, color in [(0, "#2b6cb0"), (1, "#c53030")]:
        sub = df[df[config.TARGET] == label]
        axes[0].hist(sub["tenure"], bins=30, alpha=0.6, color=color, label="Churn" if label else "Stay")
        axes[1].hist(sub["MonthlyCharges"], bins=30, alpha=0.6, color=color, label="Churn" if label else "Stay")
    axes[0].set_title("Tenure (months)"); axes[1].set_title("Monthly charges"); axes[0].legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "eda_distributions.png", dpi=150); plt.close(fig)

    (config.REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
