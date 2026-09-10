#!/usr/bin/env bash
# Regenerate every result and figure in the paper.
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/exp1_tool_routing.py
python experiments/exp2_humaid.py 12 14 16
python experiments/exp3_sequential.py
python experiments/exp4_edge.py
python experiments/exp5_uncertainty.py
python experiments/make_figures.py
python scripts/make_docx.py
echo "results/, paper/figures/ and paper/docx/ regenerated"
