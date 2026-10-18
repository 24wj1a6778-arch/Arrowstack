# Presentation Outline (8-10 slides, ~10 minutes)

1. **Title & problem** - Customer churn costs money; who should the retention team contact?
2. **Stakeholder & success metrics** - retention manager; PR-AUC, recall/precision, net business value.
3. **Data** - 7,043 customers, 20 features, 28.6% churn; simulated with the Telco schema; blank TotalCharges handled.
4. **What the data says** - `reports/figures/eda_churn_by_segment.png`: month-to-month 42% churn vs two-year 5%; fibre 40% vs 20%; new customers churn most.
5. **Method** - stratified 60/20/20 split, dummy baseline, 3 models, one pipeline, validation-only selection.
6. **Model comparison** - table from `reports/model_comparison_val.csv`; why PR-AUC; chosen model = logistic regression.
7. **Business threshold** - `threshold_vs_value.png`; threshold 0.62; contact nobody 0 / everyone -4,000 / model +6,480 on test.
8. **Test results** - PR-AUC 0.615 vs 0.286 baseline, ROC-AUC 0.814, precision 0.59, recall 0.61, lift 2.5x; confusion matrix.
9. **Error analysis** - hardest segments (new customers, month-to-month), most important features (tenure, contract, total charges), what data would help.
10. **Limitations & next steps** - simulated data, assumed costs, calibration, fairness review, monitoring; demo `predict.py` on 20 customers.

**Tip:** lead each slide with the takeaway sentence, then show the evidence. Keep the code in the repo, not on slides.
