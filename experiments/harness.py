"""
Experiment harness for TabPFN / TabICL on the time-based split.

One call = one experiment. Every run writes its config, metrics and per-loan
predictions to experiments/results/<name>/ and adds a row to results/summary.csv.

Usage (needs .venv-icl), from the project root:
    python experiments/harness.py --name 00_baseline

Fixed across runs so results are comparable: the eval rows (seeded stratified
sample of val and test), the draw seeds (SEED + draw index), and the
train/val/test files. Target-encoded features use TRAIN outcomes only
(out-of-fold for train rows).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import tabular_icl as icl  # noqa: E402  (reuse make_model, scores, segments)

DATA = ROOT / "data"
RESULTS = Path(__file__).parent / "results"
SEED = icl.SEED
SHRINK = 50   # pseudo-loans pulling each key toward the overall default rate


# ---------------------------------------------------------------- data
def target_encode(key_tr, y_tr, others, shrink=SHRINK, seed=SEED):
    """Shrunk default rate per key from TRAIN only.
    Train rows get out-of-fold values; `others` (val/test keys) use all of train.
    Returns (train_rate, [other_rates], train_logn, [other_logn])."""
    prior = y_tr.mean()

    def stats(k, y):
        g = pd.DataFrame({"k": k.values, "y": y.values}).groupby("k")["y"]
        return g.agg(n="size", d="sum")

    def apply(g, k):
        n = k.map(g["n"])
        d = k.map(g["d"])
        return ((d + shrink * prior) / (n + shrink)).values, np.log1p(n.fillna(0)).values

    tr_rate = np.full(len(key_tr), np.nan)
    tr_n = np.zeros(len(key_tr))
    for fit, fill in KFold(5, shuffle=True, random_state=seed).split(key_tr):
        r, n = apply(stats(key_tr.iloc[fit], y_tr.iloc[fit]), key_tr.iloc[fill])
        tr_rate[fill], tr_n[fill] = r, n
    full = stats(key_tr, y_tr)
    outs = [apply(full, k) for k in others]
    return tr_rate, [o[0] for o in outs], tr_n, [o[1] for o in outs]


def load_data(feature_set):
    """feature_set: base | naics4 (+ target-encoded 4-digit NAICS) |
    no_bank (drop the two bank features)."""
    parts = [pd.read_csv(DATA / f"{n}.csv", dtype={c: str for c in icl.CATEGORICAL})
             for n in ["train", "val", "test"]]
    train, val, test = parts
    features = list(icl.FEATURES)

    if feature_set == "naics4":
        naics = pd.read_csv(DATA / "sba_clean.csv", usecols=["LoanNr_ChkDgt", "NAICS"],
                            dtype={"NAICS": "string"})
        naics["naics4"] = naics["NAICS"].str[:4]
        for d in parts:
            d["naics4"] = d[["LoanNr_ChkDgt"]].merge(
                naics[["LoanNr_ChkDgt", "naics4"]], on="LoanNr_ChkDgt",
                how="left")["naics4"].values
        tr, (va, te), trn, (van, ten) = target_encode(
            train["naics4"], train["default"], [val["naics4"], test["naics4"]])
        for d, rate, n in zip(parts, (tr, va, te), (trn, van, ten)):
            d["naics4_rate"], d["naics4_logn"] = rate, n
        features += ["naics4_rate", "naics4_logn"]
    elif feature_set == "no_bank":
        features = [f for f in features if f not in ("log_bank_loans", "bank_default_rate")]

    cats = {c: sorted(train[c].dropna().unique()) for c in icl.CATEGORICAL}
    for d in parts:  # same category list everywhere; unseen -> blank
        for c in icl.CATEGORICAL:
            d[c] = pd.Categorical(d[c], categories=cats[c])
    return train, val, test, features


# ---------------------------------------------------------------- contexts
def draw_context(pool, n, seed, balance=None):
    """Sample n loans. balance=None keeps the pool's default rate;
    balance=0.5 forces 50% defaults."""
    if balance is None:
        return icl.subsample(pool, n, seed=seed)
    pos = pool[pool["default"] == 1]
    neg = pool[pool["default"] == 0]
    k = int(n * balance)
    return pd.concat([pos.sample(min(k, len(pos)), random_state=seed),
                      neg.sample(min(n - k, len(neg)), random_state=seed)])


def order_context(ctx, how, seed):
    if how == "index":      # original row order (chronological)
        return ctx.sort_index()
    return ctx.sample(frac=1, random_state=seed)   # "shuffle"


# ---------------------------------------------------------------- run
def run(a):
    out_dir = RESULTS / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    train, val, test, features = load_data(a.features)
    val = icl.subsample(val, a.eval_n)
    test = icl.subsample(test, a.eval_n)
    pool = train[train["ApprovalFY"] >= a.pool_from_fy]
    print(f"[{a.name}] {a.model} | {len(features)} features | pool {len(pool):,} "
          f"(FY>={a.pool_from_fy}) | ctx {a.ctx_size:,} x {a.draws} draws | "
          f"eval val {len(val):,} test {len(test):,}", flush=True)

    outs = {"val": val[icl.KEEP].copy(), "test": test[icl.KEEP].copy()}
    per_draw_auc = []
    for d in range(a.draws):
        t0 = time.time()
        seed = SEED + d
        ctx = order_context(draw_context(pool, a.ctx_size, seed, a.balance),
                            a.order, seed)
        model = icl.make_model(a.model, seed, a.device)
        model.fit(ctx[features], ctx["default"])
        for name, part in [("val", val), ("test", test)]:
            outs[name][f"p_draw{d}"] = icl.predict_chunked(model, part[features], a.chunk)
        auc_v = roc_auc_score(val["default"], outs["val"][f"p_draw{d}"])
        auc_t = roc_auc_score(test["default"], outs["test"][f"p_draw{d}"])
        per_draw_auc.append((auc_v, auc_t))
        print(f"  draw {d}: val AUC {auc_v:.3f}  test AUC {auc_t:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    draw_cols = [f"p_draw{d}" for d in range(a.draws)]
    rows = []
    for name, o in outs.items():
        o["p_mean"] = o[draw_cols].mean(axis=1)
        o["p_std"] = o[draw_cols].std(axis=1) if a.draws > 1 else 0.0
        o.to_csv(out_dir / f"predictions_{name}.csv", index=False)
        for seg, s in icl.segments(o, by_year=(name == "test")):
            r = icl.scores(s["default"], s["p_mean"])
            if r:
                rows.append({"data": name, "segment": seg, **r})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    pick = lambda d, s: metrics[(metrics.data == d) & (metrics.segment == s)].iloc[0]
    v, t = pick("val", "all"), pick("test", "all")
    tn, te = pick("test", "new business"), pick("test", "existing business")
    pa = np.array(per_draw_auc)
    summary = {
        "name": a.name, "model": a.model, "features": a.features,
        "n_features": len(features), "ctx_size": a.ctx_size, "draws": a.draws,
        "pool_from_fy": a.pool_from_fy, "balance": a.balance, "order": a.order,
        "eval_n": a.eval_n,
        "val_auc": v.roc_auc, "val_pr": v.pr_auc, "val_brier": v.brier,
        "val_mean_pred": v.mean_pred, "val_rate": v.default_rate,
        "test_auc": t.roc_auc, "test_pr": t.pr_auc, "test_brier": t.brier,
        "test_mean_pred": t.mean_pred, "test_rate": t.default_rate,
        "test_auc_new": tn.roc_auc, "test_auc_existing": te.roc_auc,
        "draw_auc_std_val": pa[:, 0].std(), "draw_auc_std_test": pa[:, 1].std(),
        "p_std": outs["test"]["p_std"].mean(),
        "minutes": (time.time() - t_start) / 60,
    }
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(a), "features_used": features}, indent=2, default=str))
    sfile = RESULTS / "summary.csv"
    prev = pd.read_csv(sfile) if sfile.exists() else pd.DataFrame()
    prev = prev[prev["name"] != a.name] if len(prev) else prev
    pd.concat([prev, pd.DataFrame([summary])]).to_csv(sfile, index=False)
    print(f"  => val AUC {v.roc_auc:.3f} | test AUC {t.roc_auc:.3f} "
          f"(new {tn.roc_auc:.3f}, existing {te.roc_auc:.3f}) | "
          f"test brier {t.brier:.3f} | {summary['minutes']:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="tabpfn", choices=["tabpfn", "tabicl"])
    ap.add_argument("--features", default="base", choices=["base", "naics4", "no_bank"])
    ap.add_argument("--ctx-size", type=int, default=10_000)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--eval-n", type=int, default=5_000)
    ap.add_argument("--pool-from-fy", type=int, default=2001,
                    help="draw context only from train loans with FY >= this")
    ap.add_argument("--balance", type=float, default=None,
                    help="target default share in the context (None = natural)")
    ap.add_argument("--order", default="index", choices=["index", "shuffle"])
    ap.add_argument("--chunk", type=int, default=2_000)
    ap.add_argument("--device", default="auto")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
