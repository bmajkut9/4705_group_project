"""
Clean SBAnational.csv (Li, Mickel & Taylor 2018) and write a cleaned CSV.

Usage:
    python clean_sba.py SBAnational.csv sba_clean.csv
    python clean_sba.py SBAnational.csv sba_clean.csv --min-fy 2001 --max-fy 2010

Cleaning only -- no feature engineering, no modelling.
"""
import argparse

import numpy as np
import pandas as pd

# Known only after the outcome / after approval -> removed
LEAK_COLS = ["ChgOffDate", "ChgOffPrinGr", "BalanceGross",
             "DisbursementDate", "DisbursementGross"]
MONEY_COLS = ["GrAppv", "SBA_Appv"]
INT_COLS = ["Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob",
            "FranchiseCode", "UrbanRural"]


def log(step, df):
    print(f"{step:<45} {len(df):>9,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--max-fy", type=int, default=2010,
                    help="drop loans approved after this fiscal year")
    ap.add_argument("--min-fy", type=int, default=2001,
                    help="drop loans approved before this fiscal year")
    a = ap.parse_args()

    df = pd.read_csv(a.input, dtype=str, low_memory=False)
    log("raw", df)

    # 1. Target: CHGOFF -> 1, P I F -> 0; drop rows with no outcome
    status = df["MIS_Status"].str.strip().str.upper()
    df = df[status.isin(["CHGOFF", "P I F"])].copy()
    df["default"] = (status[df.index] == "CHGOFF").astype(int)
    df = df.drop(columns=["MIS_Status"])
    log("drop missing outcome", df)

    # 2. Remove leaky columns and borrower name
    df = df.drop(columns=[c for c in LEAK_COLS + ["Name"] if c in df])

    # 3. Duplicate loan IDs
    df = df.drop_duplicates(subset="LoanNr_ChkDgt")
    log("drop duplicate loan IDs", df)

    # 4. Fix types
    for c in MONEY_COLS:  # "$60,000.00 " -> 60000.0
        df[c] = pd.to_numeric(df[c].str.replace(r"[\$,\s]", "", regex=True),
                              errors="coerce")
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # ApprovalFY has values like "1976A"
    df["ApprovalFY"] = pd.to_numeric(
        df["ApprovalFY"].str.extract(r"(\d{4})")[0], errors="coerce")
    # ApprovalDate "28-Feb-97". Two-digit years 00-68 parse as 20xx, so 1960s
    # loans (9 rows) read as 2060s; all are before FY2001 and get dropped below.
    df["ApprovalDate"] = pd.to_datetime(df["ApprovalDate"], format="%d-%b-%y",
                                        errors="coerce")

    # 5. Junk codes -> blank (NaN)
    df["NAICS"] = pd.to_numeric(df["NAICS"], errors="coerce")
    df.loc[df["NAICS"] < 100000, "NAICS"] = np.nan   # 0 = not recorded
    df["NewExist"] = df["NewExist"].where(df["NewExist"].isin([1, 2]))
    df["UrbanRural"] = df["UrbanRural"].where(df["UrbanRural"].isin([1, 2]))
    for c in ["RevLineCr", "LowDoc"]:
        v = df[c].str.strip().str.upper()
        if c == "RevLineCr":
            # "0" defaults like "N" (15.0% vs 14.6%) -> treat as No.
            # "T" (43% default, meaning unknown) and rare junk stay blank.
            v = v.replace({"0": "N"})
        df[c] = v.where(v.isin(["Y", "N"]))

    # 6. Drop rows missing fields every model needs
    df = df.dropna(subset=["ApprovalDate", "ApprovalFY", "GrAppv", "Term"])
    df = df[df["GrAppv"] > 0]
    log("drop unparseable date / amount / term", df)

    # 7. Right-censoring: recent vintages only contain early-resolved loans
    df = df[df["ApprovalFY"] <= a.max_fy]
    log(f"drop approvals after FY{a.max_fy}", df)

    # 8. Before ~FY2001, NAICS and UrbanRural are mostly unrecorded
    df = df[df["ApprovalFY"] >= a.min_fy]
    log(f"drop approvals before FY{a.min_fy}", df)

    df["NAICS"] = df["NAICS"].astype("Int64")
    for c in ["NewExist", "UrbanRural"]:
        df[c] = df[c].astype("Int64")
    df.to_csv(a.output, index=False)

    print(f"\ndefault rate: {df['default'].mean():.2%}")
    print("blank cells per column:")
    print(df.isna().sum()[lambda s: s > 0].to_string())
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()