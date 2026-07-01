# 🏦 Global Banking Stress Monitor

An institutional-grade systemic risk pipeline designed to quantify counterparty risk, systemic fragility, and capital vulnerabilities across Global Systemically Important Banks (G-SIBs).

## 🧠 Quantitative Methodologies Implemented
This engine moves beyond standard volatility modeling by aggregating four distinct dimensions of systemic risk:
1. **Network Topology (Interconnectedness):** Utilizes `NetworkX` to build a directed graph of interbank exposures sourced directly from FFIEC FR Y-9C regulatory filings, isolating bottleneck risks via PageRank and Betweenness Centrality.
2. **Conditional $\Delta$CoVaR (Spillover Risk):** Implements Adrian & Brunnermeier's (2016) quantile regression ($q=0.05$) to measure the marginal increase in systemic Value-at-Risk when a specific institution enters distress, strictly conditioned on lagged macroeconomic state variables (VIX, TED Spread, Yield Curve) to eliminate omitted variable bias.
3. **Systemic Absorption Ratio (Fragility):** A rolling 252-day Principal Component Analysis (PCA) that flags peak structural fragility when the 1st Principal Component explains >70% of cross-sectional variance.
4. **SRISK (Capital Vulnerability):** Replicates Brownlees & Engle's (2017) Long-Run Marginal Expected Shortfall ($LRMES$) to project expected capital depletion and required taxpayer bailouts during a simulated 40% systemic market collapse.

## ⚙️ Architecture & Execution
- **Data Ingestion:** Automated pull from the Federal Reserve API (FRED) and Yahoo Finance.
- **Regulatory Parser:** Custom engine to bypass legacy EOF errors and safely extract strictly-defined MDRM codes (e.g., `BHCA7206`, `P793`) from raw bulk CSVs.
- **Visualization:** Interactive `Streamlit` and `Plotly` dashboard.

### Quick Start
```powershell
# Install dependencies
pip install -r requirements.txt

# Execute the master pipeline
.\run_pipeline.ps1