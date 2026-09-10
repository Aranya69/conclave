# Build the practical guide on Windows (needs MiKTeX or TeX Live).
Set-Location $PSScriptRoot
1..3 | ForEach-Object { pdflatex -interaction=nonstopmode CONCLAVE_Guide.tex | Out-Null }
Write-Host "wrote $PSScriptRoot\CONCLAVE_Guide.pdf"
