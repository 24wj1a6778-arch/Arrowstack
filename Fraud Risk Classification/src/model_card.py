"""Generate reports/model_card.md from the saved metrics (never drifts from results).

Run:  python -m src.model_card
"""
import json
from datetime import date

import pandas as pd

from src import config


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    with open(config.REPORTS_DIR / "test_metrics.json") as f:
        t = json.load(f)
    with open(config.THRESHOLD_PATH) as f:
        info = json.load(f)
    comp = pd.read_csv(config.REPORTS_DIR / "model_comparison_val.csv")
    s, d, c = t["selected"], t["dummy_baseline"], t["strategies_cost"]
    lo, hi = t["pr_auc_95ci"]
    table = _md_table(comp[["algorithm", "strategy", "pr_auc", "roc_auc", "recall_at_precision_50"]].round(3))

    card = f"""# Model Card - Transaction Fraud Risk Classifier

*Generated {date.today().isoformat()} by `src/model_card.py`*

## Model details
- **Model:** {t['algorithm']} with imbalance strategy **{t['strategy']}**, inside a scikit-learn Pipeline (feature engineering, imputation, scaling, one-hot encoding)
- **Task:** binary classification - is a card transaction fraudulent?
- **Alert threshold:** {info['threshold']:.4f} (chosen on validation to minimise total cost)
- **Random seed:** {config.SEED}
- **Imbalance handling:** compared none / class weights / random undersampling / SMOTE (own implementation, training data only)

## Intended use
- **Stakeholder:** fraud-operations manager.
- **Decision supported:** which transactions to send for manual review.
- **Not intended for:** automatically declining customers without human review, credit decisions, or any population unlike the training data.

## Data
- Simulated card transactions (see `src/generate_data.py`): {t['n_test'] * 5:,} rows, fraud rate about {t['test_fraud_rate']:.2%}; 25% of fraud is deliberately "stealth" (looks normal).
- Stratified split 60/20/20. Test set: {t['n_test']:,} transactions, {t['n_test_frauds']} frauds.

## Performance (hold-out test set)
| Metric | Selected model | Dummy baseline |
|---|---|---|
| PR-AUC | {s['pr_auc']:.3f} (95% CI {lo:.3f}-{hi:.3f}) | {d['pr_auc']:.3f} |
| ROC-AUC | {s['roc_auc']:.3f} | {d['roc_auc']:.3f} |
| Precision | {s['precision']:.3f} | {d['precision']:.3f} |
| Recall | {s['recall']:.3f} | {d['recall']:.3f} |
| Recall at 50% precision | {s['recall_at_precision_50']:.3f} | {d['recall_at_precision_50']:.3f} |
| Alerts per 1,000 transactions | {s['alerts_per_1000']:.1f} | {d['alerts_per_1000']:.1f} |

- Confusion matrix at the chosen threshold: TP={s['tp']}, FP={s['fp']}, FN={s['fn']}, TN={s['tn']}.
- Fraud value caught: {t['fraud_amount_caught']:,.0f} of {t['fraud_amount_total']:,.0f}.

### Cost on the test set (simulated assumptions)
A missed fraud costs its amount; every alert costs {config.REVIEW_COST:.0f} in review effort.

| Strategy | Total cost |
|---|---|
| Approve everything | {c['approve_everything']:,.0f} |
| Review everything | {c['review_everything']:,.0f} |
| **Use the model** | **{c['use_model']:,.0f}** |

## Model comparison (validation set)
{table}

## Limitations & risks
- Simulated data: real fraud is adversarial, changes over time and is harder; re-validate on company data before use.
- Only ~{t['n_test_frauds']} frauds in the test set, so metrics are noisy (see the confidence interval). Differences between imbalance strategies are within this noise.
- Cost numbers (review cost, missed fraud = amount) are assumptions.
- Resampled models output scores that are not calibrated probabilities; use them for ranking and thresholds, not as literal risk percentages.
- The split is random, not time-based; production should validate on later periods.
- Stealth fraud that looks normal is missed by design; the model must not be the only control.
- Review alerts for bias across customer groups before production use.

## Next improvements
Time-based validation, behavioural features (session, device history), probability calibration, SHAP explanations, feedback loop from analyst decisions, drift monitoring, per-segment thresholds.
"""
    (config.REPORTS_DIR / "model_card.md").write_text(card)
    print(f"Wrote {config.REPORTS_DIR / 'model_card.md'}")


if __name__ == "__main__":
    main()
