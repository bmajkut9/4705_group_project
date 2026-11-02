"""
Logistic Regression and XGBoost baselines on the time-based split.

Usage:
    python baselines.py              # reads train.csv, val.csv, test.csv
    python baselines.py --weighted   # also up-weight defaults (class weighting)

Outputs:
    baseline_metrics.csv   metrics per model x dataset x segment
    test_predictions.csv   per-loan test predictions (for later evaluation)

Metrics:
    roc_auc      ranking quality (0.5 = coin flip)
    pr_auc       precision-recall AUC; a random model scores = default_rate
    brier        mean squared error of probabilities (lower = better)
    default_rate actual share of defaults in the segment
    mean_pred    average predicted probability; compare to default_rate
                 to see how far off calibration is
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TARGET = "default"
NUMERIC = ["log_loan_amount", "sba_share", "log_employees", "log_jobs_created",
           "log_jobs_retained", "log_bank_loans", "bank_default_rate"]
BINARY = ["is_new_business", "is_urban", "is_franchise", "is_revolving_line",
          "is_low_doc", "same_state_bank"]
CATEGORICAL = ["loan_size_bucket", "naics_sector", "State"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
SEED = 42


def load():
    parts = {n: pd.read_csv(f"{n}.csv", dtype={c: str for c in CATEGORICAL})
             for n in ["train", "val", "test"]}
    return parts["train"], parts["val"], parts["test"]


# --------------------------------------------------------------------------
def fit_logistic(train, weighted):
    prep = ColumnTransformer([
        # median fill + a "was blank" indicator column, then scale
        ("num", Pipeline([("imp", SimpleImputer(strategy="median",
                                                add_indicator=True)),
                          ("sc", StandardScaler())]), NUMERIC + BINARY),
        # blank becomes its own category
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]),
         CATEGORICAL),
    ])
    model = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=2000,
                                   class_weight="balanced" if weighted else None)),
    ])
    X = train[FEATURES].astype({c: "object" for c in CATEGORICAL})
    return model.fit(X, train[TARGET])


def predict_logistic(model, df):
    X = df[FEATURES].astype({c: "object" for c in CATEGORICAL})
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
def to_xgb(df, categories):
    X = df[FEATURES].copy()
    for c in CATEGORICAL:  # categories fixed from train; unseen -> blank
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def fit_xgb(train, val, weighted):
    cats = {c: sorted(train[c].dropna().astype(str).unique()) for c in CATEGORICAL}
    pos = train[TARGET].mean()
    model = XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        enable_categorical=True, tree_method="hist", max_cat_to_onehot=1,
        scale_pos_weight=(1 - pos) / pos if weighted else 1.0,
        eval_metric="aucpr", early_stopping_rounds=100,
        random_state=SEED, n_jobs=-1)
    model.fit(to_xgb(train, cats), train[TARGET],
              eval_set=[(to_xgb(val, cats), val[TARGET])], verbose=False)
    return model, cats


# --------------------------------------------------------------------------
def scores(y, p):
    if y.nunique() < 2:
        return {}
    return {"loans": len(y),
            "default_rate": y.mean(),
            "mean_pred": p.mean(),
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def segments(df):
    yield "all", df
    yield "new business", df[df["is_new_business"] == 1]
    yield "existing business", df[df["is_new_business"] == 0]
    for fy in sorted(df["ApprovalFY"].unique()):
        yield f"FY{fy}", df[df["ApprovalFY"] == fy]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weighted", action="store_true",
                    help="up-weight defaults during training")
    a = ap.parse_args()

    train, val, test = load()
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}"
          f"  weighted={a.weighted}\n")

    lr = fit_logistic(train, a.weighted)
    xgb, cats = fit_xgb(train, val, a.weighted)
    print(f"XGBoost stopped at {xgb.best_iteration + 1} trees\n")

    rows, preds = [], test[["LoanNr_ChkDgt", "ApprovalFY",
                            "is_new_business", TARGET]].copy()
    for name, d in [("val", val), ("test", test)]:
        d = d.copy()
        d["p_lr"] = predict_logistic(lr, d)
        d["p_xgb"] = xgb.predict_proba(to_xgb(d, cats))[:, 1]
        if name == "test":
            preds["p_lr"], preds["p_xgb"] = d["p_lr"].values, d["p_xgb"].values
        segs = segments(d) if name == "test" else [("all", d)]
        for seg, s in segs:
            for model, col in [("logistic", "p_lr"), ("xgboost", "p_xgb")]:
                r = scores(s[TARGET], s[col])
                if r:
                    rows.append({"model": model, "data": name,
                                 "segment": seg, **r})

    out = pd.DataFrame(rows)
    out.to_csv("baseline_metrics.csv", index=False)
    preds.to_csv("test_predictions.csv", index=False)

    show = out.set_index(["data", "segment", "model"])[
        ["loans", "default_rate", "mean_pred", "roc_auc", "pr_auc", "brier"]]
    with pd.option_context("display.width", 120):
        print(show.round(3).to_string())
    print("\nwrote baseline_metrics.csv, test_predictions.csv")


if __name__ == "__main__":
    main()