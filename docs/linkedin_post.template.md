# LinkedIn

## A. Post (paste, then add your GitHub link)

I built a credit risk scorecard from scratch, and it taught me why banks still use logistic regression.

The question: given a borrower's profile, how likely are they to be 90+ days past due within 2 years?

What I did (data: {{DATASET}}, {{N_TOTAL}} borrowers, {{DEFAULT_RATE}} default rate):
- Cleaned the data and handled placeholder codes and missing income
- Binned every variable and computed Weight of Evidence and Information Value (the most predictive variable: {{TOP_FEATURE}})
- Fit a logistic regression and converted it into a points-based credit score (600 points at 50:1 odds, +20 points doubles the odds)
- Validated on a held-out set: AUC {{AUC}}, KS {{KS}}, Gini {{GINI}}, with a train-test AUC gap of {{OVERFIT_GAP}}
- Checked stability with PSI ({{PSI}})
- Benchmarked against {{BENCH_NAME}}: the scorecard {{TRADEOFF}}

The biggest lesson: in lending, a slightly less accurate model you can explain to a customer and a regulator often beats a black box.

Everything is open source with unit tests and a notebook that walks through each step:
[GitHub link]

Next up: adding LGD and EAD to get a full Expected Loss model.

#CreditRisk #DataAnalytics #RiskAnalytics #Python #MachineLearning #Banking

---

## B. "Projects" / Featured section entry

**Title:** Credit Risk Scorecard (Probability of Default)

**Description:** End-to-end application scorecard on {{N_TOTAL}} borrowers. Implemented monotonic WoE binning, IV-based
variable selection, logistic regression and points scaling (600 at 50:1, PDO 20) in Python. Test AUC {{AUC}},
KS {{KS}}, Gini {{GINI}}; score PSI {{PSI}}. Benchmarked against {{BENCH_NAME}} (AUC {{BENCH_AUC}}). Includes unit tests and a teaching notebook.

**Skills to tag:** Credit Risk, Logistic Regression, Scorecard Development, Python, scikit-learn, Feature Engineering, Model Validation

---

## C. Short "About"/headline additions
- "Risk & Data Analytics | Credit scorecards, PD modelling, Python, SQL"
