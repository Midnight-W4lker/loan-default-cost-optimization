# Loan Default Risk with Business Cost Optimization

**Live project:** https://loan-default-cost-optimization.vercel.app

Planned models are a regularized Logistic Regression baseline and CatBoost. Model
ranking, probability calibration, and business decisions are evaluated separately.

The decision threshold is selected on validation predictions only by minimizing:

`total cost = false-negative cost × FN + false-positive cost × FP`

The default illustrative ratio is 10:1, with sensitivity analysis required because
real lending costs depend on exposure, recovery, margin, and policy. The held-out
test set is evaluated once after model and threshold selection.
