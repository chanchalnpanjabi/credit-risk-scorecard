# LinkedIn

## A. Post (paste, then add your GitHub link)

I built a credit risk scorecard from scratch, and it taught me why banks still use logistic regression.

The question: given a borrower's profile, how likely are they to be 90+ days past due within 2 years?

What I did (data: Give Me Some Credit (Kaggle), 149,999 borrowers, 6.7% default rate):
- Cleaned the data and handled placeholder codes and missing income
- Binned every variable and computed Weight of Evidence and Information Value (the most predictive variable: RevolvingUtilizationOfUnsecuredLines)
- Fit a logistic regression and converted it into a points-based credit score (600 points at 50:1 odds, +20 points doubles the odds)
- Validated on a held-out set: AUC 0.853, KS 0.55, Gini 0.71, with a train-test AUC gap of 0.000
- Checked stability with PSI (0.0003)
- Benchmarked against XGBoost: the scorecard gives up 0.015 AUC while staying fully explainable

The biggest lesson: in lending, a slightly less accurate model you can explain to a customer and a regulator often beats a black box.

Everything is open source with unit tests and a notebook that walks through each step:
[GitHub link]

Next up: adding LGD and EAD to get a full Expected Loss model.

#CreditRisk #DataAnalytics #RiskAnalytics #Python #MachineLearning #Banking

---

## B. "Projects" / Featured section entry

**Title:** Credit Risk Scorecard (Probability of Default)

**Description:** End-to-end application scorecard on 149,999 borrowers. Implemented monotonic WoE binning, IV-based
variable selection, logistic regression and points scaling (600 at 50:1, PDO 20) in Python. Test AUC 0.853,
KS 0.55, Gini 0.71; score PSI 0.0003. Benchmarked against XGBoost (AUC 0.867). Includes unit tests and a teaching notebook.

**Skills to tag:** Credit Risk, Logistic Regression, Scorecard Development, Python, scikit-learn, Feature Engineering, Model Validation

---

## C. Short "About"/headline additions
- "Risk & Data Analytics | Credit scorecards, PD modelling, Python, SQL"
