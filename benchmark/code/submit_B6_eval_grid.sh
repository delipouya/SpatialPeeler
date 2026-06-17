#!/bin/bash
# Submit B6_eval_grid.py to the LSF farm.
# Estimated runtime: ~4-6 hours (64 conditions, NMF + logistic regression + DE each).
#
# Usage:
#   bash submit_B6_eval_grid.sh
# benchmark/B6_eval_grid_615205.out

set -euo pipefail

ROOT=/lustre/scratch126/gengen/teams_v2/marks/dp31/SpatialPeeler
SCRIPT=$ROOT/benchmark/code/B6_eval_grid.py
PYTHON=$ROOT/.venv/bin/python
LOG_DIR=$ROOT/benchmark

/software/lsf-farm22/10.1/linux3.10-glibc2.17-x86_64/bin/bsub \
    -J  "B6_eval_grid" \
    -q  normal \
    -n  2 \
    -M  32000 \
    -R  "select[mem>32000] rusage[mem=32000]" \
    -o  "$LOG_DIR/B6_eval_grid_%J.out" \
    -e  "$LOG_DIR/B6_eval_grid_%J.err" \
    "$PYTHON $SCRIPT"

echo "Submitted. Monitor with: bjobs"
echo "Logs: $LOG_DIR/B6_eval_grid_<JOBID>.out"
echo "Results: $ROOT/benchmark/benchmark_results_grid_v5_64parameters.csv"
