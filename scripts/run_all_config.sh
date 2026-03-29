#!/bin/bash
# MSBD5021 Assignment 1: Demonstrate REINFORCE for all (n, T) combinations
# Usage: bash scripts/run_all_config.sh

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "MSBD5021 Assignment 1: Multi-Asset Allocation with REINFORCE"
echo "============================================================"
echo ""

for n in 3 4; do
    for T in 1 2 3 4 5 6 7 8 9; do
        echo ">>> Running n=${n}, T=${T} ..."
        uv run python asset_alloc.py "configs/n${n}_T${T}.json" -o "results/training_n${n}_T${T}.png"
        echo ""
    done
done

echo "============================================================"
echo "All 18 configurations completed."
echo "Training curves saved as results/training_n{n}_T{T}.png"
echo "============================================================"
