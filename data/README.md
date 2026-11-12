# Data

Raw data is not committed to this repo.

## Real data (use this for your results)
1. Go to Kaggle, search **"Give Me Some Credit"**, open the competition and accept the rules.
2. Download `cs-training.csv` and put it in this folder: `data/cs-training.csv`.

150,000 borrowers, 10 features, target `SeriousDlqin2yrs` (1 = 90+ days past due within 2 years).

## Synthetic stand-in (for testing only)
```bash
python -m src.make_synthetic --out data/synthetic_training.csv
```
Same columns and the same quirks (missing income, 96/98 codes, utilisation > 1, age 0) but randomly generated.
Never quote metrics from it as real results.
