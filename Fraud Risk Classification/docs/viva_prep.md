# Viva / Interview Preparation - Project 2

Use these as starting points; answer in your own words.

**What problem does it solve and how did you define success?**
Fraud analysts can review only a few transactions. Success = catch as much fraud value as possible at a manageable alert volume, measured by PR-AUC, recall at fixed precision, alerts per 1,000 and total cost.

**Why not use accuracy?**
98.75% of transactions are legitimate, so a model that never flags anything is 98.75% accurate and useless.

**How did you handle the imbalance?**
Compared four strategies - none, class weights, random undersampling, SMOTE - for three algorithms. Resampling happens only on training data inside `fit`; validation and test stay untouched.

**What is SMOTE and what is its risk?**
It creates synthetic frauds by interpolating between real ones. Risks: it can create unrealistic points (especially with one-hot columns), shifts the score scale (not calibrated probabilities), and must never be applied before splitting.

**Which decision had the biggest impact?**
Choosing the threshold by cost rather than 0.5, and working in alert budgets. Scores from resampled models are not probabilities, so 0.5 would have been meaningless.

**Precision vs recall - what is the trade-off here?**
More alerts catch more fraud but waste more reviews. At 1% alerts precision is about 0.78 with recall about 0.62; at 5% precision falls to about 0.18 while recall barely rises to 0.73.

**Did SMOTE win?**
It scored highest on validation, but by a margin (0.009) inside noise with 150 frauds. I would not claim it is better; I would say imbalance handling made little difference on this data.

**How did you explain predictions?**
Global: permutation importance. Local: what-if attribution - replace a feature with typical values and see how far the score falls. SHAP is a next step.

**What are the limits and risks?**
Simulated data, few test frauds, assumed costs, random not time-based split, stealth fraud missed, scores not calibrated, bias review needed.

**What would change at 10x the data?**
Time-based validation, gradient boosting at scale, feature store, streaming scoring, drift monitoring, analyst feedback loop.

**Explain it to a non-technical manager.**
"The model gives every transaction a risk score. We review the riskiest 1-2%. In testing that costs about 4,100 versus 24,500 if we ignore fraud, and it catches about 86% of the fraud value, but it will miss fraud that looks completely normal."

**Terms:** PR-AUC, precision, recall, class imbalance, SMOTE, undersampling, class weights, threshold, operating point, alert budget, data leakage, permutation importance, bootstrap interval, model card.
