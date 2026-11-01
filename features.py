"""
Build model features from the cleaned SBA file.

Usage:
    python build_features.py sba_clean.csv sba_features.csv

Input:  output of clean_sba.py
Output: one row per loan with
          - ID / split columns (NOT features): LoanNr_ChkDgt, ApprovalDate, ApprovalFY
          - target: default
          - the feature columns listed in FEATURES below
Blank cells are kept blank (NaN). XGBoost and TabPFN handle them natively;
the logistic regression pipeline will impute them later.

Deliberately excluded:
  - Term, and anything derived from it (real-estate / recession flags):
    shown to be leaky (single-column ROC-AUC 0.945; 9% round terms for
    defaults vs 83% for paid loans).
  - Bank-level features (e.g. bank size / bank default rate): must be
    computed on the training split only, so they belong in the split step.
  - City, Zip: too many distinct values; geography is covered by State.
"""
import sys

import numpy as np
import pandas as pd

ID_COLS = ["LoanNr_ChkDgt", "ApprovalDate", "ApprovalFY"]
TARGET = "default"

NUMERIC = ["log_loan_amount", "sba_share", "log_employees",
           "log_jobs_created", "log_jobs_retained"]
BINARY = ["is_new_business", "is_urban", "is_franchise",
          "is_revolving_line", "is_low_doc", "same_state_bank"]
CATEGORICAL = ["loan_size_bucket", "naics_sector", "State"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Loan-size buckets at SBA program thresholds:
# 150k = LowDoc cap / guarantee-rate break, 350k = SBAExpress cap
SIZE_BINS = [0, 50_000, 150_000, 350_000, 1_000_000, np.inf]
SIZE_LABELS = ["<=50k", "50k-150k", "150k-350k", "350k-1M", ">1M"]

# 2-digit NAICS -> sector code (multi-code sectors grouped)
SECTOR_GROUPS = {"31": "31-33", "32": "31-33", "33": "31-33",
                 "44": "44-45", "45": "44-45", "48": "48-49", "49": "48-49"}
VALID_SECTORS = {"11", "21", "22", "23", "31-33", "42", "44-45", "48-49",
                 "51", "52", "53", "54", "55", "56", "61", "62", "71", "72",
                 "81", "92"}


def yes_no(s):
    return s.map({"Y": 1, "N": 0})  # anything else stays blank


def main():
    src, dst = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src, dtype={"NAICS": "string"})
    out = df[ID_COLS + [TARGET]].copy()

    # Loan size
    out["log_loan_amount"] = np.log(df["GrAppv"])
    out["loan_size_bucket"] = pd.cut(df["GrAppv"], SIZE_BINS,
                                     labels=SIZE_LABELS, right=True)
    out["sba_share"] = (df["SBA_Appv"] / df["GrAppv"]).clip(0, 1)

    # Business size / jobs (log1p tames the long right tail)
    out["log_employees"] = np.log1p(df["NoEmp"].clip(lower=0))
    out["log_jobs_created"] = np.log1p(df["CreateJob"].clip(lower=0))
    out["log_jobs_retained"] = np.log1p(df["RetainedJob"].clip(lower=0))

    # Flags (blank stays blank)
    out["is_new_business"] = df["NewExist"].map({1: 0, 2: 1})
    out["is_urban"] = df["UrbanRural"].map({1: 1, 2: 0})
    out["is_franchise"] = (df["FranchiseCode"] > 1).astype(int)  # 0/1 = none
    out["is_revolving_line"] = yes_no(df["RevLineCr"])
    out["is_low_doc"] = yes_no(df["LowDoc"])
    out["same_state_bank"] = np.where(
        df["State"].isna() | df["BankState"].isna(), np.nan,
        (df["State"] == df["BankState"]).astype(float))

    # Industry: 2-digit NAICS sector, blank if NAICS was blank
    two = df["NAICS"].str[:2].replace(SECTOR_GROUPS)
    out["naics_sector"] = two.where(two.isin(VALID_SECTORS))

    out["State"] = df["State"]

    out = out[ID_COLS + [TARGET] + FEATURES]
    out.to_csv(dst, index=False)

    print(f"{len(out):,} loans, {len(FEATURES)} features -> {dst}\n")
    print("blank share per feature:")
    print(out[FEATURES].isna().mean().round(3)[lambda s: s > 0].to_string())
    print("\ndefault rate by loan size:")
    print(out.groupby("loan_size_bucket", observed=True)[TARGET]
             .agg(loans="size", default_rate="mean").round(3).to_string())


if __name__ == "__main__":
    main()