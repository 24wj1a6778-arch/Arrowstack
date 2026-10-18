# Decision Log

| # | Decision | Alternatives considered | Reason |
|---|---|---|---|
| 1 | Use simulated data with the public Telco schema | Download the real Telco CSV | Handbook allows simulated data; keeps the project runnable offline. Real file is a drop-in replacement. |
| 2 | Stratified 60/20/20 split, test set touched once | Plain random split; k-fold only | Stratification preserves the ~29% churn rate in every split; a separate validation set allows model *and* threshold selection without leaking the test set. |
| 3 | Select models on PR-AUC | Accuracy, ROC-AUC | Churn is the minority class; accuracy hides poor recall; PR-AUC focuses on the positive class. |
| 4 | Dummy baseline first | Start with a complex model | Gives a floor every real model must beat and a fair "is this useful?" reference. |
| 5 | One sklearn `Pipeline` including feature engineering | Preprocess in a notebook, then train | Same code at train and predict time, no training/serving skew, no leakage (imputers/scalers fit on train only). |
| 6 | Threshold chosen on validation by business value | Default 0.5; maximise F1 | Offers have a cost and a benefit; the best business threshold (0.62) differs from 0.5. |
| 7 | Keep the model trained on train only (no refit on train+val) | Refit on train+val | Keeps the validated model and its threshold exactly consistent; trade-off is slightly less training data. |
| 8 | Permutation importance on raw columns | Model coefficients; SHAP | Model-agnostic, works through the pipeline, no extra dependency. SHAP listed as an extension. |
| 9 | Error-analysis ranking needs >= 20 churners per segment | Rank all segments | First run flagged "Two year contract: 100% missed" from a handful of churners - noise, not insight. |
| 10 | Tests with `unittest` | pytest | No extra dependency; pytest also runs them. |

*Add your own entries as you work - mentors read this log.*
