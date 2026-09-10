#!/usr/bin/env bash
# Build the practical guide. Three passes so the table of contents settles.
set -euo pipefail
cd "$(dirname "$0")"
for i in 1 2 3; do
  pdflatex -interaction=nonstopmode CONCLAVE_Guide.tex > /dev/null
done
echo "wrote $(pwd)/CONCLAVE_Guide.pdf ($(pdfinfo CONCLAVE_Guide.pdf | awk '/Pages/{print $2}') pages)"
