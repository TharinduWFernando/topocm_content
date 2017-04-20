#!/bin/bash
export MPLCONFIGDIR=$(mktemp -d)
export  IPYTHONDIR=$(mktemp -d)
echo "c.HistoryManager.enabled = False" > $IPYTHONDIR/ipython_config.py
jupyter nbconvert --to notebook --inplace --config scripts/config_filter_smudge.py $1
