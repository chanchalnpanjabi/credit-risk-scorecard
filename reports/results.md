# Results  
*Data source: **Give Me Some Credit (Kaggle)***

- Borrowers: 104,999 train / 45,000 test; default rate 6.68%
- **Test AUC 0.853** (95% CI 0.845 to 0.860), **KS 0.554**, **Gini 0.705**
- Train AUC 0.852 vs test 0.853 (gap -0.000; small gap = no overfitting)
- Integer-points scorecard test AUC 0.852 (rounding to whole points costs essentially nothing)
- XGBoost benchmark test AUC 0.867
- Score PSI train vs test 0.0003 (<0.10 = stable)
- Scale: 600 points at 50:1 odds, +20 points doubles the odds; test score range 415 to 619
- Variables in final model: RevolvingUtilizationOfUnsecuredLines, NumberOfTimes90DaysLate, NumberOfTime30-59DaysPastDueNotWorse, NumberOfTime60-89DaysPastDueNotWorse, age, MonthlyIncome, NumberOfDependents

## Information value

| feature                              |    iv | strength            |
|:-------------------------------------|------:|:--------------------|
| RevolvingUtilizationOfUnsecuredLines | 1.117 | suspiciously strong |
| NumberOfTimes90DaysLate              | 0.852 | suspiciously strong |
| NumberOfTime30-59DaysPastDueNotWorse | 0.739 | suspiciously strong |
| NumberOfTime60-89DaysPastDueNotWorse | 0.577 | suspiciously strong |
| age                                  | 0.239 | medium              |
| NumberOfOpenCreditLinesAndLoans      | 0.091 | weak                |
| MonthlyIncome                        | 0.066 | weak                |
| NumberRealEstateLoansOrLines         | 0.039 | weak                |
| NumberOfDependents                   | 0.036 | weak                |
| DebtRatio                            | 0.019 | useless             |

## Score bands (test)

|   band |    n |   min_score |   max_score |   bads |   bad_rate |   share_of_all_bads |   cum_share_of_bads |
|-------:|-----:|------------:|------------:|-------:|-----------:|--------------------:|--------------------:|
|      1 | 4500 |         415 |         539 |   1635 |     0.3633 |              0.5436 |              0.5436 |
|      2 | 4500 |         539 |         558 |    486 |     0.108  |              0.1616 |              0.7051 |
|      3 | 4500 |         558 |         572 |    313 |     0.0696 |              0.1041 |              0.8092 |
|      4 | 4500 |         572 |         583 |    180 |     0.04   |              0.0598 |              0.869  |
|      5 | 4500 |         583 |         593 |    128 |     0.0284 |              0.0426 |              0.9116 |
|      6 | 4500 |         593 |         598 |     85 |     0.0189 |              0.0283 |              0.9398 |
|      7 | 4500 |         598 |         602 |     71 |     0.0158 |              0.0236 |              0.9634 |
|      8 | 4500 |         602 |         608 |     53 |     0.0118 |              0.0176 |              0.9811 |
|      9 | 4500 |         608 |         612 |     30 |     0.0067 |              0.01   |              0.991  |
|     10 | 4500 |         612 |         619 |     27 |     0.006  |              0.009  |              1      |

## Cut-off strategy (test)

|   cutoff |   approval_rate |   bad_rate_among_approved |   bads_caught_by_rejecting |   overall_bad_rate |
|---------:|----------------:|--------------------------:|---------------------------:|-------------------:|
|      415 |          1      |                    0.0668 |                     0      |             0.0668 |
|      539 |          0.9009 |                    0.0341 |                     0.5402 |             0.0668 |
|      558 |          0.8058 |                    0.025  |                     0.6988 |             0.0668 |
|      572 |          0.7038 |                    0.0182 |                     0.8078 |             0.0668 |
|      583 |          0.6022 |                    0.0147 |                     0.8677 |             0.0668 |
|      593 |          0.5104 |                    0.0119 |                     0.9092 |             0.0668 |

## Selection log

- drop DebtRatio: IV < 0.02
- WARNING RevolvingUtilizationOfUnsecuredLines: IV 1.12 > 0.5 (check for leakage)
- WARNING NumberOfTimes90DaysLate: IV 0.85 > 0.5 (check for leakage)
- WARNING NumberOfTime30-59DaysPastDueNotWorse: IV 0.74 > 0.5 (check for leakage)
- WARNING NumberOfTime60-89DaysPastDueNotWorse: IV 0.58 > 0.5 (check for leakage)
- drop NumberRealEstateLoansOrLines: positive coefficient (sign flip from collinearity)
- drop NumberOfOpenCreditLinesAndLoans: positive coefficient (sign flip from collinearity)