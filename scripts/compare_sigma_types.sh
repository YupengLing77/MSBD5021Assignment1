#!/bin/bash
# MSBD5021 Assignment 1: Compare three sigma types on n4_T9.json
# Usage: bash scripts/compare_sigma_types.sh

set -e
cd "$(dirname "$0")/.."

CONFIG="configs/n4_T9.json"
OUTDIR="results"
mkdir -p "$OUTDIR"

for SIGMA_TYPE in fixed manual learn; do
    echo "=== Running sigma type: $SIGMA_TYPE ==="
    uv run python asset_alloc.py "$CONFIG" --policy-sigma-type $SIGMA_TYPE -o "$OUTDIR/curve_n4_T9_${SIGMA_TYPE}.png"
    echo "Output: $OUTDIR/curve_n4_T9_${SIGMA_TYPE}.png"
done

echo "All sigma type runs completed."
