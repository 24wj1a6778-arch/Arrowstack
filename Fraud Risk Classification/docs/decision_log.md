# Decision Log

| # | Decision | Alternatives | Reason |
|---|---|---|---|
| 1 | Simulated transactions with readable features | Kaggle credit-card fraud (V1-V28) | Handbook allows simulated data; readable features make explanations meaningful. |
| 2 | 25% of fraud is "stealth" | All fraud detectable | Realistic; gives a genuine error-analysis story and keeps results honest. |
| 3 | Stratified 60/20/20, test used once | Time-based split | Keeps parity with Project 1 and enough frauds per split; time-based validation listed as a next step. |
| 4 | Select on validation PR-AUC | Accuracy, ROC-AUC | Fraud is ~1% of data; accuracy is meaningless. PR-AUC focuses on the rare class. |
| 5 | Compare none / class weights / undersampling / SMOTE | Pick one technique up front | The point is to test whether imbalance handling helps; results show differences within noise. |
| 6 | Resampling inside the model's `fit` | Resample the whole dataset first | Prevents leakage: validation and test are never resampled, and the same code works in the pipeline. |
| 7 | Implement undersampling and SMOTE ourselves | imbalanced-learn | Removes a dependency that is hard to install in some environments; tested in `tests/`. imblearn is a documented alternative. |
| 8 | Cost-based threshold (missed fraud = amount, alert = 5) | Default 0.5, maximise F1 | Fraud losses depend on the amount; cost directly encodes the business trade-off. |
| 9 | Report alert-budget operating points | One threshold only | Fraud teams plan by "alerts they can review"; gives the manager choices. |
| 10 | Permutation importance + what-if attributions | SHAP | Works through the pipeline with no extra install; SHAP is optional per the handbook and a next step. |
| 11 | Bootstrap CI for test PR-AUC | Single number | Only 150 frauds in test; a single number overstates certainty. |
| 12 | Risk band and decision from the same threshold | Fixed 0.2/0.5 bands (Project 1) | Lesson from Project 1: bands that disagree with the decision confuse users. |

*Add your own entries as you work - mentors read this log.*
