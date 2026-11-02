# SBA loan default prediction

Predict whether an SBA-guaranteed loan is charged off (`default` = 1) or paid in full, using only
information known at approval. Compares logistic regression, XGBoost, and the tabular in-context
learners TabPFN and TabICL on a time-based split.

- Pipeline reasoning (leakage checks, cleaning, features): `Pipeline_Notes.md`
- Experiments and results: `Experiments_Notes.md` and `experiments/README.md`

## Data

The dataset is not in the repo (171 MB, over GitHub's limit). Download `SBAnational.csv` from
https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied
and put it at `archive/SBAnational.csv`.

## Setup

The virtual environments are not committed. There are two, because the tabular models need a newer Python.

```bash
# 1. cleaning, features, split, XGBoost / logistic baselines (Python 3.9+)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. TabPFN / TabICL and the experiments (Python 3.10+)
python3.12 -m venv .venv-icl
.venv-icl/bin/pip install -r requirements-icl.txt
```

TabPFN on CPU is capped at 5,000 context rows, so use a GPU (CUDA or Apple Silicon `mps`) for
the 10,000-row runs used here.

## Run order

Scripts read and write the current folder, and all generated files go in `data/` (git-ignored):

```bash
mkdir -p data && cd data
../.venv/bin/python ../code/clean_sba.py ../archive/SBAnational.csv sba_clean.csv
../.venv/bin/python ../code/features.py sba_clean.csv sba_features.csv
../.venv/bin/python ../code/split_data.py sba_features.csv sba_clean.csv   # train/val/test.csv
../.venv/bin/python ../code/baselines.py                                    # logistic + XGBoost
../.venv-icl/bin/python ../code/tabular_icl.py --model tabpfn               # optional, GPU
cd ..
zsh experiments/run_all.sh                                                  # experiments (uses .venv-icl)
```

Leakiness check (single-column XGBoost AUCs): `../.venv/bin/python ../code/leakiness_testing.py sba_clean.csv`

## Layout

| Path | What |
|---|---|
| `code/` | cleaning, features, split, baselines, tabular ICL, leakiness test |
| `experiments/` | harness, run script, per-run outputs in `results/`, summary in `README.md` |
| `data/`, `archive/` | generated and raw data (not committed) |
