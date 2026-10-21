# Model Card - Transaction Fraud Risk Classifier

*Generated 2026-10-19 by `src/model_card.py`*

## Model details
- **Model:** random_forest with imbalance strategy **smote**, inside a scikit-learn Pipeline (feature engineering, imputation, scaling, one-hot encoding)
- **Task:** binary classification - is a card transaction fraudulent?
- **Alert threshold:** 0.2754 (chosen on validation to minimise total cost)
- **Random seed:** 42
- **Imbalance handling:** compared none / class weights / random undersampling / SMOTE (own implementation, training data only)

## Intended use
- **Stakeholder:** fraud-operations manager.
- **Decision supported:** which transactions to send for manual review.
- **Not intended for:** automatically declining customers without human review, credit decisions, or any population unlike the training data.

## Data
- Simulated card transactions (see `src/generate_data.py`): 60,000 rows, fraud rate about 1.25%; 25% of fraud is deliberately "stealth" (looks normal).
- Stratified split 60/20/20. Test set: 12,000 transactions, 150 frauds.

## Performance (hold-out test set)
| Metric | Selected model | Dummy baseline |
|---|---|---|
| PR-AUC | 0.648 (95% CI 0.578-0.719) | 0.013 |
| ROC-AUC | 0.853 | 0.500 |
| Precision | 0.599 | 0.000 |
| Recall | 0.627 | 0.000 |
| Recall at 50% precision | 0.647 | 0.000 |
| Alerts per 1,000 transactions | 13.1 | 0.0 |

- Confusion matrix at the chosen threshold: TP=94, FP=63, FN=56, TN=11787.
- Fraud value caught: 21,189 of 24,504.

### Cost on the test set (simulated assumptions)
A missed fraud costs its amount; every alert costs 5 in review effort.

| Strategy | Total cost |
|---|---|
| Approve everything | 24,504 |
| Review everything | 60,000 |
| **Use the model** | **4,099** |

## Model comparison (validation set)
| algorithm | strategy | pr_auc | roc_auc | recall_at_precision_50 |
|---|---|---|---|---|
| dummy_baseline | none | 0.012 | 0.500 | 0.000 |
| logistic_regression | none | 0.653 | 0.873 | 0.653 |
| logistic_regression | class_weight | 0.628 | 0.874 | 0.633 |
| logistic_regression | undersample | 0.640 | 0.875 | 0.647 |
| logistic_regression | smote | 0.643 | 0.876 | 0.640 |
| random_forest | none | 0.657 | 0.861 | 0.673 |
| random_forest | class_weight | 0.645 | 0.868 | 0.667 |
| random_forest | undersample | 0.630 | 0.870 | 0.640 |
| random_forest | smote | 0.673 | 0.867 | 0.680 |
| hist_gradient_boosting | none | 0.655 | 0.868 | 0.673 |
| hist_gradient_boosting | class_weight | 0.615 | 0.873 | 0.613 |
| hist_gradient_boosting | undersample | 0.636 | 0.869 | 0.653 |
| hist_gradient_boosting | smote | 0.664 | 0.885 | 0.653 |

## Limitations & risks
- Simulated data: real fraud is adversarial, changes over time and is harder; re-validate on company data before use.
- Only ~150 frauds in the test set, so metrics are noisy (see the confidence interval). Differences between imbalance strategies are within this noise.
- Cost numbers (review cost, missed fraud = amount) are assumptions.
- Resampled models output scores that are not calibrated probabilities; use them for ranking and thresholds, not as literal risk percentages.
- The split is random, not time-based; production should validate on later periods.
- Stealth fraud that looks normal is missed by design; the model must not be the only control.
- Review alerts for bias across customer groups before production use.

## Next improvements
Time-based validation, behavioural features (session, device history), probability calibration, SHAP explanations, feedback loop from analyst decisions, drift monitoring, per-segment thresholds.
