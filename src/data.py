"""Loading, cleaning and splitting. Every cleaning rule is logged so it can be defended."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "SeriousDlqin2yrs"
DELINQ_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SPECIAL_CODES = (96, 98)  # known error / placeholder codes in the delinquency counts


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    return df.drop(columns=unnamed)


def clean(df: pd.DataFrame):
    """Return (clean_df, log).

    Rules (and why):
      1. drop rows with no target or age < 18  -> cannot be a valid borrower
      2. 96 / 98 in delinquency counts -> NaN  -> they are codes, not "96 times late";
         NaN later gets its own 'Missing' bin so their (very high) risk is still captured
      3. NO capping of outliers: tree-based binning puts extreme values in the last bin,
         so they cannot distort the model. (One reason scorecards bin everything.)
    """
    log = {"rows_in": int(len(df))}
    df = df.copy()

    bad = df[TARGET].isna() | (df["age"] < 18)
    log["dropped_invalid_target_or_age"] = int(bad.sum())
    df = df.loc[~bad]

    n_codes = 0
    for c in DELINQ_COLS:
        m = df[c].isin(SPECIAL_CODES)
        n_codes += int(m.sum())
        df.loc[m, c] = np.nan
    log["delinquency_special_codes_set_to_nan"] = n_codes

    log["rows_out"] = int(len(df))
    log["default_rate"] = float(df[TARGET].mean())
    return df.reset_index(drop=True), log


def split(df: pd.DataFrame, test_size: float = 0.3, seed: int = 42):
    """Stratified split. Binning/WoE are fitted on TRAIN ONLY (no leakage)."""
    tr, te = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
