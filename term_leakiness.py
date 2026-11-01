"""
Is Term leaky? Train XGBoost on Term alone and see how predictive it is.

Usage:
    python check_term.py sba_clean.csv
"""
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv(sys.argv[1])
y = df["default"]

# Other approval-time columns, for comparison (numeric only, no encoding)
OTHER = ["GrAppv", "SBA_Appv", "NoEmp", "NewExist", "CreateJob",
         "RetainedJob", "UrbanRural"]


def auc(cols, train_idx, test_idx):
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                      random_state=0, n_jobs=-1)
    m.fit(df.loc[train_idx, cols], y[train_idx])
    return roc_auc_score(y[test_idx], m.predict_proba(df.loc[test_idx, cols])[:, 1])


splits = {
    "random 80/20": train_test_split(df.index, test_size=0.2,
                                     stratify=y, random_state=0),
    "time (train FY<=2003, test FY2004+)": (
        df.index[df["ApprovalFY"] <= 2003], df.index[df["ApprovalFY"] > 2003]),
}

for name, (tr, te) in splits.items():
    print(f"\n{name}")
    print(f"  Term only           ROC-AUC = {auc(['Term'], tr, te):.3f}")
    print(f"  other cols, no Term ROC-AUC = {auc(OTHER, tr, te):.3f}")
    print(f"  other cols + Term   ROC-AUC = {auc(OTHER + ['Term'], tr, te):.3f}")

# Default rate by term length -- look for a cliff rather than a smooth trend
bins = [-1, 12, 36, 60, 84, 120, 180, 239, 1000]
labels = ["<=1y", "1-3y", "3-5y", "5-7y", "7-10y", "10-15y", "15-20y", "20y+"]
tbl = (df.assign(term=pd.cut(df["Term"], bins, labels=labels))
         .groupby("term", observed=True)["default"]
         .agg(loans="size", default_rate="mean"))
print("\nDefault rate by term:")
print(tbl.round(3).to_string())