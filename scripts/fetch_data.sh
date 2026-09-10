#!/usr/bin/env bash
# Fetch every dataset used by the CONCLAVE experiments into ./data
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/bfcl data/humaid

echo "==> Berkeley Function Calling Leaderboard v3"
BASE="https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
for f in BFCL_v3_simple BFCL_v3_multiple BFCL_v3_live_simple BFCL_v3_live_multiple \
         BFCL_v3_irrelevance BFCL_v3_live_irrelevance BFCL_v3_parallel \
         BFCL_v3_parallel_multiple; do
  curl -sL -o "data/bfcl/$f.json" "$BASE/$f.json"
done
for f in BFCL_v3_simple BFCL_v3_multiple BFCL_v3_live_simple BFCL_v3_live_multiple; do
  curl -sL -o "data/bfcl/ans_$f.json" "$BASE/possible_answer/$f.json"
done

echo "==> HumAID (QCRI), official splits"
for s in train validation test; do
  curl -sL -o "data/humaid/$s.parquet" \
    "https://huggingface.co/api/datasets/QCRI/HumAID-all/parquet/default/$s/0.parquet"
done

echo "==> AI4I 2020 predictive maintenance (UCI, optional)"
curl -sL -o data/ai4i2020.csv \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv" || true

echo "==> NSL-KDD (optional)"
curl -sL -o data/nsl_kdd_train.txt \
  "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B_20Percent.txt" || true

echo "done. Datasets are used under their respective licences; cite the original papers."
