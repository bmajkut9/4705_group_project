#!/bin/zsh
# The experiments reported in experiments/README.md. From the project root:
#   zsh experiments/run_all.sh
PY=.venv-icl/bin/python
H=experiments/harness.py

$PY $H --name 00_baseline                            # random context, 10k, 3 draws
$PY $H --name 10_recent_fy2003 --pool-from-fy 2003   # context only from recent years
$PY $H --name 11_recent_fy2004 --pool-from-fy 2004
$PY $H --name 12_balanced_50   --balance 0.5         # 50% defaults in context
$PY $H --name 20_order_shuffle --order shuffle       # same context, shuffled order
$PY $H --name 30_ctx_2000      --ctx-size 2000       # context size
$PY $H --name 31_ctx_20000     --ctx-size 20000
$PY $H --name 40_naics4        --features naics4     # target-encoded 4-digit NAICS
$PY $H --name 41_no_bank       --features no_bank    # do the bank features help?
$PY $H --name 50_tabicl        --model tabicl        # other model
