# ==============================================================================
# Global Banking Stress Monitor - Master Execution Pipeline
# ==============================================================================

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " 🏦 INITIATING GLOBAL BANKING STRESS MONITOR PIPELINE" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Ensure we are in the correct directory
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# [1/8] Check Virtual Environment
Write-Host "`n[1/8] Activating Virtual Environment..." -ForegroundColor Yellow
if (-Not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[!] Virtual environment not found. Please run 'python -m venv .venv' first." -ForegroundColor Red
    exit
}

# [2/8] Fetching Macro State Variables (FRED)
Write-Host "`n[2/8] Fetching Macro State Variables (FRED)..." -ForegroundColor Yellow
& .venv\Scripts\python.exe src/market_data_pipeline.py
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Pipeline halted at step 2." -ForegroundColor Red; exit }

# [3/8] Parsing FR Y-9C Regulatory Tape
Write-Host "`n[3/8] Parsing FR Y-9C Regulatory Tape..." -ForegroundColor Yellow
& .venv\Scripts\python.exe src/data_ingestion.py
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Pipeline halted at step 3." -ForegroundColor Red; exit }

# [4/8 & 5/8] Network Construction & Centrality
Write-Host "`n[4/8 & 5/8] Constructing Interbank Network & Calculating Centrality..." -ForegroundColor Yellow
& .venv\Scripts\python.exe src/dynamic_network.py
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Pipeline halted at steps 4/5." -ForegroundColor Red; exit }

# [6/8 & 7/8] Risk Metrics (Absorption Ratio, CoVaR, SRISK)
Write-Host "`n[6/8 & 7/8] Calculating PCA Absorption Ratio & Tail-Risk Regressions..." -ForegroundColor Yellow
& .venv\Scripts\python.exe src/data_pipeline.py
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Pipeline halted at steps 6/7." -ForegroundColor Red; exit }

# [8/8] PIDL Walk-Forward Training
Write-Host "`n[8/8] Executing Phase 3: PIDL Walk-Forward Training..." -ForegroundColor Yellow
& .venv\Scripts\python.exe src/train_pidl.py
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Training failed. Halting pipeline." -ForegroundColor Red; exit }

# [Final] Inference & Visualisation
Write-Host "`n[Final] Executing Phase 4: Inference & Visualisation..." -ForegroundColor Yellow
& .venv\Scripts\python.exe src/visualization.py
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Visualisation failed." -ForegroundColor Red; exit }

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host " ✅ PIPELINE COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host " Dashboard saved to: reports/figures/systemic_risk_dashboard.png" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green

# [Launch] Open Live Streamlit Web Dashboard
Write-Host "`n[Dashboard] Launching Live Streamlit Dashboard in Web Browser..." -ForegroundColor Cyan
Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m streamlit run src/dashboard.py"