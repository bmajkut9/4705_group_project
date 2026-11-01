"""
TabPFN / TabICL predictions on the time-based split, using sampled context.

Usage:
    python tabular_icl.py --model tabpfn                 # quick run, defaults below
    python tabular_icl.py --model tabicl
    python tabular_icl.py --model tabpfn --eval-sample 0 --context-size 50000  # full run (GPU)

How it works:
    There is no training. For each of --draws rounds, a random context of
    --context-size TRAIN loans is drawn (stratified on default, so it keeps the
    train default rate), the model is "fit" on that context, and it predicts
    every val and test loan. Predictions are averaged over the draws.
    Per-draw predictions are kept: their spread is the context-sampling
    noise RQ2 is about.

Defaults are sized for a laptop dry run:
    --context-size 10000   loans per context draw
    --draws 5              context draws to average
    --eval-sample 20000    val/test loans scored (stratified, fixed seed);
                           0 = score all of val and test
Requires Python >= 3.10. TabPFN on CPU is capped at 5,000 context rows,
so use a GPU (CUDA, or Apple Silicon "mps").

Outputs:
    {model}_val_predictions.csv, {model}_test_predictions.csv
        LoanNr_ChkDgt, ApprovalFY, is_new_business, default,
        p_draw0..p_drawK, p_mean, p_std
    {model}_contexts.csv   which train loans were in each context draw
    {model}_metrics.csv    same segments as baselines.py
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

TARGET = "default"
NUMERIC = ["log_loan_amount", "sba_share", "log_employees", "log_jobs_created",
           "log_jobs_retained", "log_bank_loans", "bank_default_rate"]
BINARY = ["is_new_business", "is_urban", "is_franchise", "is_revolving_line",
          "is_low_doc", "same_state_bank"]
CATEGORICAL = ["loan_size_bucket", "naics_sector", "State"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
KEEP = ["LoanNr_ChkDgt", "ApprovalFY", "is_new_business", TARGET]
SEED = 42


def make_model(name, seed, device):
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        # fit_with_cache: encode the context once, reuse it for every chunk
        return TabPFNClassifier(device=device, random_state=seed,
                                fit_mode="fit_with_cache")
    if name == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=None if device == "auto" else device,
                                random_state=seed, kv_cache=True)
    raise ValueError(name)


def load():
    parts = [pd.read_csv(f"{n}.csv", dtype={c: str for c in CATEGORICAL})
             for n in ["train", "val", "test"]]
    train = parts[0]
    cats = {c: sorted(train[c].dropna().unique()) for c in CATEGORICAL}
    for d in parts:  # same category list everywhere; unseen -> blank
        for c in CATEGORICAL:
            d[c] = pd.Categorical(d[c], categories=cats[c])
    return parts


def subsample(df, n, seed=SEED):
    if n <= 0 or n >= len(df):
        return df
    keep, _ = train_test_split(df, train_size=n, stratify=df[TARGET],
                               random_state=seed)
    return keep.sort_index()


def predict_chunked(model, X, chunk):
    return np.concatenate([model.predict_proba(X.iloc[i:i + chunk])[:, 1]
                           for i in range(0, len(X), chunk)])


def scores(y, p):
    if y.nunique() < 2:
        return None
    return {"loans": len(y), "default_rate": y.mean(), "mean_pred": p.mean(),
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def segments(df, by_year):
    yield "all", df
    if by_year:
        yield "new business", df[df["is_new_business"] == 1]
        yield "existing business", df[df["is_new_business"] == 0]
        for fy in sorted(df["ApprovalFY"].unique()):
            yield f"FY{fy}", df[df["ApprovalFY"] == fy]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tabpfn", "tabicl"], required=True)
    ap.add_argument("--context-size", type=int, default=10_000)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--eval-sample", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=2_000,
                    help="rows predicted per call (lower if out of memory)")
    ap.add_argument("--device", default="auto",
                    help="auto, cuda, mps or cpu")
    a = ap.parse_args()

    train, val, test = load()
    val, test = subsample(val, a.eval_sample), subsample(test, a.eval_sample)
    print(f"{a.model}: context {a.context_size:,} x {a.draws} draws | "
          f"scoring val {len(val):,}, test {len(test):,}\n")

    outs = {"val": val[KEEP].copy(), "test": test[KEEP].copy()}
    contexts = []
    for d in range(a.draws):
        t0 = time.time()
        ctx = subsample(train, a.context_size, seed=SEED + d)
        contexts.append(pd.DataFrame({"draw": d,
                                      "LoanNr_ChkDgt": ctx["LoanNr_ChkDgt"]}))
        model = make_model(a.model, SEED + d, a.device)
        model.fit(ctx[FEATURES], ctx[TARGET])
        for name, part in [("val", val), ("test", test)]:
            outs[name][f"p_draw{d}"] = predict_chunked(model, part[FEATURES],
                                                       a.chunk)
        auc = roc_auc_score(test[TARGET], outs["test"][f"p_draw{d}"])
        print(f"  draw {d}: test ROC-AUC {auc:.3f}  ({time.time() - t0:.0f}s)")

    rows = []
    draw_cols = [f"p_draw{d}" for d in range(a.draws)]
    for name, o in outs.items():
        o["p_mean"] = o[draw_cols].mean(axis=1)
        o["p_std"] = o[draw_cols].std(axis=1) if a.draws > 1 else 0.0
        o.to_csv(f"{a.model}_{name}_predictions.csv", index=False)
        for seg, s in segments(o, by_year=(name == "test")):
            r = scores(s[TARGET], s["p_mean"])
            if r:
                rows.append({"model": a.model, "data": name, "segment": seg, **r})

    pd.concat(contexts).to_csv(f"{a.model}_contexts.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(f"{a.model}_metrics.csv", index=False)

    with pd.option_context("display.width", 120):
        print("\n" + out.set_index(["data", "segment"]).drop(columns="model")
              .round(3).to_string())
    if a.draws > 1:
        per_draw = [roc_auc_score(test[TARGET], outs["test"][c]) for c in draw_cols]
        print(f"\ncontext-sampling noise on test: ROC-AUC across draws "
              f"{np.mean(per_draw):.3f} +/- {np.std(per_draw):.3f}; "
              f"average per-loan prediction std {outs['test']['p_std'].mean():.3f}")
    print(f"\nwrote {a.model}_val_predictions.csv, {a.model}_test_predictions.csv, "
          f"{a.model}_contexts.csv, {a.model}_metrics.csv")


if __name__ == "__main__":
    main()