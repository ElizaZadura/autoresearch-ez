#!/usr/bin/env bash
# WARMDOWN_RATIO sweep at TIME_BUDGET = 7200s (2h) per run.
# Four runs back-to-back, each with its own labelled output directory.
#
# Usage: ./run_warmdown_sweep.sh
# Logs:  output/<label>/console.log
# Total: ~8h on this machine.
#
# Each run respects:
#   AUTORESEARCH_WARMDOWN_RATIO  - overrides WARMDOWN_RATIO in train.py
#   AUTORESEARCH_RUN_LABEL       - overrides the auto-generated run dir name

set -u  # treat unset variables as errors; do NOT set -e (we want to continue on a single-run failure)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter not found or not executable: $PYTHON" >&2
    echo "Override with PYTHON=... if your venv is elsewhere." >&2
    exit 1
fi
DATE_TAG="$(date +%Y-%m-%d)"
SWEEP_DIR="output/${DATE_TAG}_warmdown_sweep_2h"
mkdir -p "$SWEEP_DIR"
SUMMARY="$SWEEP_DIR/summary.txt"

RATIOS=(0.30 0.50 0.70 0.90)

{
    echo "WARMDOWN_RATIO sweep — started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Repo:    $REPO_ROOT"
    echo "Python:  $PYTHON"
    echo "Ratios:  ${RATIOS[*]}"
    echo "Per-run budget: 7200s (2h)"
    echo "----------------------------------------------------------------"
} | tee "$SUMMARY"

sweep_t0=$(date +%s)

for ratio in "${RATIOS[@]}"; do
    label="${DATE_TAG}_2h_warmdown_${ratio}_run"
    run_dir="output/${label}"
    log_path="${run_dir}/console.log"
    mkdir -p "$run_dir"

    {
        echo
        echo "================================================================"
        echo "Run: WARMDOWN_RATIO=${ratio}    label=${label}"
        echo "Started:  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Log file: ${log_path}"
        echo "----------------------------------------------------------------"
    } | tee -a "$SUMMARY"

    run_t0=$(date +%s)

    AUTORESEARCH_WARMDOWN_RATIO="$ratio" \
    AUTORESEARCH_RUN_LABEL="$label" \
        "$PYTHON" -u train.py >"$log_path" 2>&1
    rc=$?

    run_t1=$(date +%s)
    elapsed=$((run_t1 - run_t0))

    {
        echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')   exit=${rc}   elapsed=${elapsed}s"
    } | tee -a "$SUMMARY"

    if [[ $rc -ne 0 ]]; then
        echo "WARNING: run with WARMDOWN_RATIO=${ratio} exited non-zero (${rc}); continuing." \
            | tee -a "$SUMMARY"
    fi
done

sweep_t1=$(date +%s)
total=$((sweep_t1 - sweep_t0))

{
    echo
    echo "================================================================"
    echo "Sweep done — $(date '+%Y-%m-%d %H:%M:%S')   total=${total}s"
    echo "Run dirs:"
    for ratio in "${RATIOS[@]}"; do
        label="${DATE_TAG}_2h_warmdown_${ratio}_run"
        echo "  output/${label}"
    done
} | tee -a "$SUMMARY"
