"""Generate reports/model_card.md from the saved metrics (so it never drifts
from the actual results).

Run:  python -m src.model_card
"""
import json
from datetime import date

import pandas as pd

from src import config


def main() -> None:
    with open(config.REPORTS_DIR / "test_metrics.json") as f:
        t = json.load(f)
    with open(config.THRESHOLD_PATH) as f:
        info = json.load(f)
    comp = pd.read_csv(config.REPORTS_DIR / "model_comparison_val.csv")
    s, d = t["selected"], t["dummy_baseline"]
    strat = t["strategies"]

    card = f"""# Model Card - Customer Churn Predictor

*Generated {date.today().isoformat()} by `src/model_card.py`*

## Model details
- **Model:** {t['model']} inside a scikit-learn Pipeline (feature engineering + imputation + scaling + one-hot encoding)
- **Task:** binary classification - will a telecom customer churn?
- **Decision threshold:** {info['threshold']:.2f} (chosen on the validation set to maximise business value)
- **Random seed:** {config.SEED}

## Intended use
- **Stakeholder:** customer-retention team.
- **Decision supported:** which customers receive a retention offer this month.
- **Not intended for:** pricing decisions, credit decisions, or any use on populations unlike the training data.

## Data
- Simulated telecom data with the public Telco-churn schema (see `src/generate_data.py`), or the real public Telco file if placed in `data/raw/`.
- Stratified split: 60% train / 20% validation / 20% test. Test churn rate: {t['test_churn_rate']:.1%}.

## Performance (hold-out test set, n = {t['n_test']:,})
| Metric | Selected model | Dummy baseline |
|---|---|---|
| PR-AUC | {s['pr_auc']:.3f} | {d['pr_auc']:.3f} |
| ROC-AUC | {s['roc_auc']:.3f} | {d['roc_auc']:.3f} |
| Precision | {s['precision']:.3f} | {d['precision']:.3f} |
| Recall | {s['recall']:.3f} | {d['recall']:.3f} |
| F1 | {s['f1']:.3f} | {d['f1']:.3f} |

- Top-decile lift: **{t['top_decile_lift']:.2f}x** (the 10% highest-risk customers contain {t['top_decile_lift']:.1f}x more churners than average).
- Confusion matrix at the chosen threshold: TP={s['tp']}, FP={s['fp']}, FN={s['fn']}, TN={s['tn']}.

### Business value on the test set (simulated assumptions)
Offer cost {config.OFFER_COST:.0f} per contact; {config.ACCEPT_RATE:.0%} of contacted churners accept; each saved customer is worth {config.SAVE_VALUE:.0f}.

| Strategy | Net value |
|---|---|
| Contact nobody | {strat['contact_nobody']:,.0f} |
| Contact everyone | {strat['contact_everyone']:,.0f} |
| **Use the model** | **{strat['use_model']:,.0f}** |

## Model comparison (validation set, threshold 0.5)
{_md_table(comp[['model', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1']])}

## Limitations & risks
- Trained on simulated/public data: real-world performance will differ; re-validate on company data before use.
- Business-value numbers depend on assumed offer cost, acceptance rate and customer value.
- Probabilities may be mis-calibrated (see `reports/figures/calibration.png`); do not read them as exact risk.
- The model finds correlation, not cause: contacting a customer will not help if the real cause is, for example, network quality.
- Possible fairness concern: features such as SeniorCitizen and gender should be reviewed before any production use.
- Churn behaviour drifts over time; monitor performance and retrain regularly.

## Next improvements
Add behavioural features (support tickets, usage drops), time-based validation, probability calibration, and monitoring of precision/recall in production.
"""
    (config.REPORTS_DIR / "model_card.md").write_text(card)
    print(f"Wrote {config.REPORTS_DIR / 'model_card.md'}")


def _md_table(df: pd.DataFrame) -> str:
    """Tiny markdown table writer (avoids needing the `tabulate` package)."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    main()
