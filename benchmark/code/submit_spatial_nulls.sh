#!/bin/bash
# Submit spatial_nulls_benchmark.py to the LSF farm.
# Estimated runtime: ~3–4 hours.
#
# Usage:
#   bash submit_spatial_nulls.sh
#(SpatialPeeler) dp31@farm22-head1:/lustre/scratch126/gengen/teams_v2/marks/dp31/SpatialPeeler/benchmark/code$ bash submit_spatial_nulls.sh
#Job <819024> is submitted to queue <normal>.
#Submitted. Monitor with: bjobs
#Logs will appear in: /lustre/scratch126/gengen/teams_v2/marks/dp31/SpatialPeeler/benchmark/spatial_nulls_results/job_<JOBID>.out
#Results will be saved to: /lustre/scratch126/gengen/teams_v2/marks/dp31/SpatialPeeler/benchmark/spatial_nulls_results/

set -euo pipefail

ROOT=/lustre/scratch126/gengen/teams_v2/marks/dp31/SpatialPeeler
SCRIPT=$ROOT/benchmark/code/spatial_nulls_benchmark.py
PYTHON=$ROOT/.venv/bin/python
LOG_DIR=$ROOT/benchmark/spatial_nulls_results
mkdir -p "$LOG_DIR"

/software/lsf-farm22/10.1/linux3.10-glibc2.17-x86_64/bin/bsub \
    -J  "spatial_nulls_benchmark" \
    -q  normal \
    -n  2 \
    -M  32000 \
    -R  "select[mem>32000] rusage[mem=32000]" \
    -o  "$LOG_DIR/job_%J.out" \
    -e  "$LOG_DIR/job_%J.err" \
    "$PYTHON $SCRIPT"

echo "Submitted. Monitor with: bjobs"
echo "Logs will appear in: $LOG_DIR/job_<JOBID>.out"
echo "Results will be saved to: $LOG_DIR/"
