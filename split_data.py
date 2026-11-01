"""
Split loans by approval year and add bank features computed from TRAIN only.

Usage:
    python split_data.py sba_features.csv sba_clean.csv

Writes train.csv, val.csv, test.csv to the current folder.

Split (by ApprovalFY):
    train  FY2001-2004   (pre-crisis)
    val    FY2005        (tuning / thresholds)
    test   FY2006-2010   (crisis-era vintages)

Bank features (Bank name comes from sba_clean.csv, joined on loan ID):
    log_bank_loans     log(1 + number of TRAIN loans made by this bank)
    bank_default_rate  bank's TRAIN default rate, shrunk toward the overall
                       train rate so small banks don't get extreme values.
                       Train rows use out-of-fold values (5 folds) so a loan's
                       own outcome never feeds its own feature.
                       Banks never seen in train -> blank.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

TRAIN_YEARS = (2001, 2004)
VAL_YEARS = (2005, 2005)
TEST_YEARS = (2006, 2010)
SHRINK = 50   # pseudo-loans pulling each bank toward the overall rate
SEED = 42


def bank_stats(train):
    g = train.groupby("Bank")["default"].agg(n="size", d="sum")
    prior = train["default"].mean()
    rate = (g["d"] + SHRINK * prior) / (g["n"] + SHRINK)
    return g["n"], rate


def main():
    feat_path = sys.argv[1] if len(sys.argv) > 1 else "sba_features.csv"
    clean_path = sys.argv[2] if len(sys.argv) > 2 else "sba_clean.csv"

    df = pd.read_csv(feat_path)
    banks = pd.read_csv(clean_path, usecols=["LoanNr_ChkDgt", "Bank"])
    df = df.merge(banks, on="LoanNr_ChkDgt", how="left")

    fy = df["ApprovalFY"]
    train = df[fy.between(*TRAIN_YEARS)].copy()
    val = df[fy.between(*VAL_YEARS)].copy()
    test = df[fy.between(*TEST_YEARS)].copy()

    # Val / test: stats from all of train
    n, rate = bank_stats(train)
    for part in (val, test):
        part["log_bank_loans"] = np.log1p(part["Bank"].map(n).fillna(0))
        part["bank_default_rate"] = part["Bank"].map(rate)

    # Train: out-of-fold stats
    train["log_bank_loans"] = np.log1p(train["Bank"].map(n).fillna(0))
    train["bank_default_rate"] = np.nan
    for fit_idx, fill_idx in KFold(5, shuffle=True, random_state=SEED).split(train):
        _, r = bank_stats(train.iloc[fit_idx])
        train.iloc[fill_idx, train.columns.get_loc("bank_default_rate")] = (
            train["Bank"].iloc[fill_idx].map(r).values)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.drop(columns="Bank").to_csv(f"{name}.csv", index=False)
        unseen = part["bank_default_rate"].isna().mean()
        new = part["is_new_business"].eq(1).mean()
        print(f"{name:<5} {int(part['ApprovalFY'].min())}-{int(part['ApprovalFY'].max())}"
              f"  {len(part):>8,} loans  default {part['default'].mean():.1%}"
              f"  new-business {new:.1%}  bank unseen in train {unseen:.1%}")


if __name__ == "__main__":
    main()