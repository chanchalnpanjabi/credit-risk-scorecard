# Interview prep: Credit Risk Scorecard

Practise answering these out loud. Keep each answer under 60 seconds. Then open the notebook and be ready to
walk through any step.

## The 30-second pitch
"I built an application scorecard that predicts the probability a borrower goes 90 days past due within two years.
I binned each variable, computed Weight of Evidence, fit a logistic regression and scaled it to a 600-point score
where 20 points doubles the odds. I validated it with AUC, KS and Gini on a held-out set, checked calibration and
PSI stability, and benchmarked it against gradient boosting to show the cost of interpretability."

## Concepts

**1. What is your default definition and why?**
90+ days past due within 2 years, the same threshold banks use for non-performing loans. It comes with the dataset, and
I would always state it first because every metric depends on it.

**2. Why not just use accuracy?**
Only a few percent of borrowers default, so predicting "nobody defaults" is about 93% accurate and useless. I use
AUC, KS and Gini, which measure ranking power.

**3. What is Weight of Evidence?**
`ln(%good / %bad)` for a bin. Positive means safer than average, negative riskier, zero means no information. It puts
all variables on a log-odds scale, so a logistic regression on WoE is a clean additive model, and it also handles
missing values and outliers through binning.

**4. What is Information Value and how did you use it?**
The sum over bins of `(%good - %bad) * WoE`. I used it to rank variables and dropped anything below 0.02. Above about
0.5 I would suspect leakage.

**5. Why logistic regression and not XGBoost?**
Explainability and regulation: every attribute maps to points, so each decline can be justified (adverse action
reasons). My benchmark shows the accuracy given up is small. If a challenger model were much better I would
investigate why (usually interactions) before replacing the scorecard.

**6. Why did you bin continuous variables? Doesn't that lose information?**
Slightly, but it gives monotonic, explainable relationships, robustness to outliers, a natural home for missing
values, and captures non-linearity inside a linear model.

**7. How did you handle missing values?**
Missing is its own bin, so the model learns its risk instead of me guessing an imputation. Missing income can itself
be informative.

**8. What were the data quality problems?**
Age of 0, utilisation far above 1, and 96/98 in the delinquency columns, which are codes rather than counts, so I set
them to missing so they got their own bin. Check the real data for the DebtRatio behaviour when income is missing.

**9. How did you avoid data leakage?**
Split first, then fitted bins, WoE, selection and coefficients on train only.

**10. Why must the coefficients be negative?**
I model P(default) and WoE increases with safety, so higher WoE should lower the log-odds of default. A positive sign
signals multicollinearity, so the code drops that variable and refits.

**11. Explain the score scaling.**
`Score = Offset + Factor * ln(odds_good)`, Factor = PDO/ln 2, Offset = 600 - Factor * ln(50). So 600 points means
50:1 good:bad odds and every +20 points doubles the odds. Since the model is additive, I split it into integer points
per bin; I checked that rounding to integers did not change AUC.

**12. Explain AUC, KS, Gini.**
AUC: chance a random bad is scored riskier than a random good. KS: maximum separation between the cumulative bad and
good distributions. Gini = 2*AUC - 1.

**13. What is PSI and what thresholds do you use?**
Population Stability Index compares the score (or variable) distribution now against the development sample:
below 0.10 stable, 0.10 to 0.25 investigate, above 0.25 significant shift and re-validate. Mine compares train and test, so it is a
sanity check; in production I would run it monthly.

**14. Is the model calibrated?**
I compared mean predicted PD with the observed default rate in each score decile (see `reports/calibration_test.csv`).
Ranking and calibration are different properties; calibration matters when PD feeds pricing or provisioning.

**15. How would you pick a cut-off?**
Use the cut-off table: approval rate versus bad rate among approved. Then choose using the profit per good loan and
loss per bad loan (cost-sensitive), not a default-rate target alone.

## Hard questions (answer honestly)

**What are the limitations?** Reject inference (only approved borrowers have outcomes), no out-of-time validation,
age is used although regulators restrict it in some jurisdictions, no LGD/EAD yet.

**A borrower with a low PD defaulted. Is the model wrong?**
No. A scorecard ranks risk, and a PD of 2.4% still means about 1 in 40 such borrowers default. (My notebook scores one
applicant by hand and shows exactly this.) Judge the model on groups, via calibration and decile bad rates, not on
single cases.

**What would you do with more time?** LGD and EAD for Expected Loss, out-of-time validation, challenger models,
monotonic gradient boosting as a benchmark, and a Streamlit app serving the points table.

**Where does this connect to your internship?** At Metis I fitted distributions to banking and portfolio data;
this is the retail-credit counterpart: PD modelling and validation.
