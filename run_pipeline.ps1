# ==============================================================================
# GLOBAL BANKING STRESS MONITOR - MASTER EXECUTION PIPELINE
# ==============================================================================

Write-Host "🚀 INITIALIZING QUANTITATIVE PIPELINE..." -ForegroundColor Cyan

# 1. Activate Virtual Environment
Write-Host "[1/8] Activating Virtual Environment..." -ForegroundColor Yellow
& ..\.venv\Scripts\Activate.ps1

# 2. Data Engineering: Macro Factors
Write-Host "[2/8] Fetching Macro State Variables (FRED)..." -ForegroundColor Yellow
python src/build_macro_factors.py
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Pipeline failed at Macro Factors." -ForegroundColor Red; exit }

# 3. Data Engineering: Regulatory Parsing
Write-Host "[3/8] Parsing FR Y-9C Regulatory Tape..." -ForegroundColor Yellow
python src/parse_fry9c_bulk.py
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Pipeline failed at Regulatory Parsing." -ForegroundColor Red; exit }

# 4. Data Engineering: Network Edges
Write-Host "[4/8] Constructing Interbank Baseline Network..." -ForegroundColor Yellow
python src/build_baseline_network.py
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Pipeline failed at Network Construction." -ForegroundColor Red; exit }

# 5. Risk Engine: Topology
Write-Host "[5/8] Calculating Network Centrality..." -ForegroundColor Yellow
python src/network_analysis.py
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Pipeline failed at Network Analysis." -ForegroundColor Red; exit }

# 6. Risk Engine: Systemic Fragility
Write-Host "[6/8] Calculating PCA Absorption Ratio..." -ForegroundColor Yellow
python src/calculate_absorption.py
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Pipeline failed at PCA Absorption." -ForegroundColor Red; exit }

# 7. Risk Engine: Conditional CoVaR & SRISK
Write-Host "[7/8] Executing Tail-Risk Regressions (CoVaR & SRISK)..." -ForegroundColor Yellow
python src/risk_metrics.py
python src/calculate_srisk.py
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Pipeline failed at Risk Regressions." -ForegroundColor Red; exit }

# 8. Launch Interface
Write-Host "[8/8] Launching Streamlit Dashboard..." -ForegroundColor Green
streamlit run src/dashboard.py