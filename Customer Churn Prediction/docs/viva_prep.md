# Viva / Interview Preparation - Project 1

Use these as starting points and put the answers in your own words.

**What business problem does it solve and how did you define success?**
The retention team can only afford to contact some customers. Success = better than contacting everyone or no one, measured by PR-AUC plus net business value after offer costs.

**Which decision had the biggest impact?**
Choosing the threshold by business value (0.62 instead of 0.5), and selecting on PR-AUC rather than accuracy because only ~29% of customers churn.

**How did you validate the work?**
Stratified train/validation/test split, test used once, dummy baseline, fixed seed, 18 unit tests, segment-level error analysis.

**Why logistic regression won, and is that trustworthy?**
Logistic regression scored best on validation PR-AUC. The data here is simulated from a logistic process, so that is expected; on real data I would re-run the comparison and not assume the same winner.

**Edge cases and risks?**
Blank TotalCharges, missing columns, negative values, unseen categories, class imbalance, fairness of SeniorCitizen/gender, assumed business costs.

**What if the data were 10x bigger?**
Move to gradient boosting/LightGBM, store data in Parquet or a database, schedule the pipeline, track experiments with MLflow, monitor drift.

**Explain it to a non-technical manager.**
"The model scores every customer by how likely they are to leave. We contact only the high-scoring ones. In our test, that earns money, while contacting everybody loses it."

**Where did you need independent problem-solving?**
Examples from this project: handling blank TotalCharges, noisy segment statistics in error analysis, picking a business-driven threshold. Add your own.

**Terms to be ready to define:** PR-AUC, ROC-AUC, precision, recall, F1, class imbalance, data leakage, threshold, calibration, lift, permutation importance, model card.
