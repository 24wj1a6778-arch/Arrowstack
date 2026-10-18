# Model Card - Customer Churn Predictor

*Generated 2026-10-18 by `src/model_card.py`*

## Model details
- **Model:** logistic_regression inside a scikit-learn Pipeline (feature engineering + imputation + scaling + one-hot encoding)
- **Task:** binary classification - will a telecom customer churn?
- **Decision threshold:** 0.62 (chosen on the validation set to maximise business value)
- **Random seed:** 42

## Intended use
- **Stakeholder:** customer-retention team.
- **Decision supported:** which customers receive a retention offer this month.
- **Not intended for:** pricing decisions, credit decisions, or any use on populations unlike the training data.

## Data
- Simulated telecom data with the public Telco-churn schema (see `src/generate_data.py`), or the real public Telco file if placed in `data/raw/`.
- Stratified split: 60% train / 20% validation / 20% test. Test churn rate: 28.6%.

## Performance (hold-out test set, n = 1,409)
| Metric | Selected model | Dummy baseline |
|---|---|---|
| PR-AUC | 0.615 | 0.286 |
| ROC-AUC | 0.814 | 0.500 |
| Precision | 0.592 | 0.000 |
| Recall | 0.613 | 0.000 |
| F1 | 0.602 | 0.000 |

- Top-decile lift: **2.52x** (the 10% highest-risk customers contain 2.5x more churners than average).
- Confusion matrix at the chosen threshold: TP=247, FP=170, FN=156, TN=836.

### Business value on the test set (simulated assumptions)
Offer cost 20 per contact; 30% of contacted churners accept; each saved customer is worth 200.

| Strategy | Net value |
|---|---|
| Contact nobody | 0 |
| Contact everyone | -4,000 |
| **Use the model** | **6,480** |

## Model comparison (validation set, threshold 0.5)
| model | pr_auc | roc_auc | precision | recall | f1 |
|---|---|---|---|---|---|
| dummy_baseline | 0.286 | 0.500 | 0.000 | 0.000 | 0.000 |
| logistic_regression | 0.583 | 0.790 | 0.468 | 0.787 | 0.587 |
| random_forest | 0.566 | 0.783 | 0.524 | 0.665 | 0.586 |
| hist_gradient_boosting | 0.564 | 0.777 | 0.589 | 0.452 | 0.511 |

## Limitations & risks
- Trained on simulated/public data: real-world performance will differ; re-validate on company data before use.
- Business-value numbers depend on assumed offer cost, acceptance rate and customer value.
- Probabilities may be mis-calibrated (see `reports/figures/calibration.png`); do not read them as exact risk.
- The model finds correlation, not cause: contacting a customer will not help if the real cause is, for example, network quality.
- Possible fairness concern: features such as SeniorCitizen and gender should be reviewed before any production use.
- Churn behaviour drifts over time; monitor performance and retrain regularly.

## Next improvements
Add behavioural features (support tickets, usage drops), time-based validation, probability calibration, and monitoring of precision/recall in production.
