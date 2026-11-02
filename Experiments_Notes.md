# Tabular ICL experiments

Goal: with the pipeline built, try a few changes to TabPFN / TabICL and see what moves the results.

## Setup (same for every run unless the row says otherwise)

- Model: TabPFN, 16 features (the pipeline's features incl. bank features), `.venv-icl`, Apple GPU
- Context: 10,000 random train loans (FY2001-2004), stratified on default, 3 draws averaged
- Scored on the same fixed random sample of 5,000 val loans (FY2005) and 5,000 test loans (FY2006-2010)
- Target-encoded features use train outcomes only (out-of-fold for train rows)
- Run with `zsh experiments/run_all.sh` from the project root. Code: `harness.py`. Each run saves `config.json`, `metrics.csv` and per-loan predictions in `results/<name>/`; `results/summary.csv` has one row per run.

For reference, the full-data XGBoost baseline (`baselines.py`, all 222,794 test loans) scores test AUC 0.689 and val AUC 0.721. Those are not on the same 5,000-loan sample, so they are only a rough comparison.

## Results

| Run | What changed | Val AUC | Test AUC | Test AUC new biz | Test Brier | Test mean pred |
|---|---|---|---|---|---|---|
| 00_baseline | nothing | 0.734 | 0.695 | 0.651 | 0.245 | 0.169 |
| 10_recent_fy2003 | context only from FY2003-2004 | 0.736 | 0.693 | 0.643 | 0.245 | 0.167 |
| 11_recent_fy2004 | context only from FY2004 | 0.741 | 0.693 | 0.659 | 0.239 | 0.180 |
| 12_balanced_50 | context is 50% defaults | 0.737 | 0.678 | 0.647 | 0.221 | 0.455 |
| 20_order_shuffle | same context, shuffled row order | 0.734 | 0.695 | 0.651 | 0.245 | 0.169 |
| 30_ctx_2000 | context size 2,000 | 0.704 | 0.678 | 0.626 | 0.248 | 0.168 |
| 31_ctx_20000 | context size 20,000 | 0.740 | 0.700 | 0.667 | 0.243 | 0.170 |
| 40_naics4 | + target-encoded 4-digit NAICS (2 features) | 0.738 | 0.701 | 0.661 | 0.241 | 0.175 |
| 41_no_bank | drop the 2 bank features | 0.699 | 0.682 | 0.647 | 0.253 | 0.155 |
| 50_tabicl | TabICL instead of TabPFN | 0.729 | 0.690 | 0.647 | 0.255 | 0.149 |

Actual default rate: val 25.3%, test 35.8%.

## How to read this (noise)

Across the 3 draws of one run, test AUC varies by about 0.005-0.007 (standard deviation), and a 5,000-loan eval sample adds more uncertainty. **Differences under about 0.01 AUC are not distinguishable from noise.** I did not run significance tests, so the findings below are only the big, consistent effects.

## What worked

- **Bank features help.** Removing them drops val AUC 0.734 -> 0.699 and test 0.695 -> 0.682. This is the clearest result, and it's the bigger gap on val.
- **A small context hurts.** 2,000 loans: val 0.704, test 0.678. Going to 20,000 gives a small gain (test 0.700, val 0.740) at 2x the runtime, which is within noise on its own.
- **Balanced context fixes calibration, not ranking.** The baseline predicts an average 17% default when the real test rate is 36%. A 50/50 context moves the average prediction to 46% and Brier improves 0.245 -> 0.221, but AUC falls to 0.678. Since AUC only depends on ranking, the change in AUC means balancing also changed how loans are ordered, and not for the better.

## What didn't (or can't tell)

- **Recent-years context** (FY2003-04 or FY2004 only): no change in AUC (0.693 vs 0.695). My guess was that matching the crisis-era vintages would help. It didn't, at least not at this noise level.
- **Row order:** shuffled results are identical to the baseline to three decimals. TabPFN may already be insensitive to order (it permutes internally), but I haven't confirmed that in its code, so I can't say why.
- **4-digit NAICS encoding:** +0.006 test, +0.004 val. In the right direction on both, but inside the noise. Not a clear win.
- **TabICL:** slightly below TabPFN (0.690 vs 0.695 test), with worse Brier.

## Takeaways

1. The model's main weakness is calibration under the crisis shift (predicts ~17% vs 36% actual), not ranking. Every variant is stuck near AUC 0.68-0.70, and XGBoost is around 0.69.
2. Context size and the bank features matter. Fancy context selection and row order don't.
3. Balancing the context is a quick calibration fix, but it costs ranking quality.


