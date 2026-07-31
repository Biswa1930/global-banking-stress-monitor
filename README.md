# 🏦 Global Banking Stress Monitor (GBSM)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Quant Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-005660)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Pipeline](https://img.shields.io/badge/Pipeline-8%2F8%20Stages%20Passing-brightgreen)

## 📌 Executive Summary

The **Global Banking Stress Monitor** is an institutional-grade, end-to-end quantitative pipeline that measures, models, and visualizes systemic risk across Global Systemically Important Banks (G-SIBs).

Rather than relying on isolated metrics like standard volatility, the engine aggregates four distinct dimensions of financial fragility — **Topological Interconnectedness, Systemic Spillovers, Market Fragility, and Capital Vulnerability** — into a single, unified risk picture. It programmatically ingests messy regulatory filings (FFIEC FR Y-9C) and live market data, runs the full quantitative stack, and renders the results on an interactive dashboard.

The pipeline is now fully operational end-to-end: a single orchestrator script takes you from raw data ingestion to a live Streamlit dashboard in one run, with no manual data wrangling required.

---

## ✅ Pipeline Status

All 8 stages of `run_pipeline.ps1` execute cleanly, start to finish:

| Stage | Component | Status |
|-------|-----------|--------|
| 1/8 | Virtual environment activation | ✅ Complete |
| 2/8 | Macro Factor Engine (FRED ingestion) | ✅ Complete — 710 trading days built |
| 3/8 | FR Y-9C Regulatory Tape Parser | ✅ Complete — 40 bank-quarter records compiled (2019–2023) |
| 4/8 | Baseline Interbank Network Construction | ✅ Complete |
| 5/8 | Network Centrality (PageRank / Gravity Model) | ✅ Complete — 1,560 edges generated across 40 global hubs |
| 6/8 | Systemic Absorption Ratio (Rolling PCA) | ✅ Complete — 458 rolling windows |
| 7/8 | Tail-Risk Regressions (ΔCoVaR & SRISK) | ✅ Complete — SRISK computed for 7 of 8 US G-SIBs |
| 8/8 | Streamlit Dashboard | ✅ Complete — live at `localhost:8501` |

**Known data gap:** BK (Bank of New York Mellon) market cap could not be resolved via the Yahoo Finance quote endpoint during the last run (`404 Quote not found`), so SRISK is currently reported for 7 of the 8 US G-SIBs. This is a live-data availability issue rather than a pipeline defect — a ticker/data-source fallback is the natural next hardening step.

---

## 🧠 The Mathematics of Systemic Risk (A Conceptual Guide)

Financial crises don't happen because a single bank makes one bad trade — they happen because banks are heavily interconnected. This project uses financial mathematics to measure how a shock to one institution ripples through the global economy.

### 1. ΔCoVaR (Systemic Spillover Risk)
Traditional Value-at-Risk (VaR) treats banks like isolated islands, measuring how much a single bank might lose in a worst case. **CoVaR** (Adrian & Brunnermeier, 2016) instead measures how much the *entire financial system* stands to lose when one specific bank is in distress. If VaR measures the risk of a single house burning down, ΔCoVaR measures the risk that the fire spreads to the whole neighborhood. The engine estimates this via 5th-percentile **Quantile Regression** conditioned on macro factors (VIX, yield curve).

In the latest run, the most vulnerable banks by ΔCoVaR were **NABZY, BNS, and BMO**.

### 2. SRISK (Capital Vulnerability)
While ΔCoVaR looks outward (damage a bank *causes*), **SRISK** (Brownlees & Engle, 2017) looks inward (damage a bank *takes*) — the expected capital shortfall a bank would face if global equities collapsed 40% over six months:

$$SRISK_i = k \cdot Debt_i - (1 - k) \cdot Equity_i \cdot (1 - LRMES_i)$$

It's effectively the "bailout bill." In the latest run, **Citigroup (C)** topped the list with an estimated SRISK of **$91.3B**, followed by **BAC ($40.6B)** and **WFC ($34.8B)**.

### 3. Systemic Absorption Ratio (Market Fragility)
During a panic, diversification disappears — everything sells off together. The engine runs a 252-day rolling **Principal Component Analysis (PCA)** over bank-return covariances to measure what share of market movement is driven by a single systemic factor. Above ~70%, idiosyncratic bank traits stop mattering and the market is primed for a cascading move. The most recent run flagged **late-October 2025** as the period of maximum fragility, with the Absorption Ratio peaking above **0.83**.

### 4. CET1 & Regulatory Network Topology
**CET1 (Common Equity Tier 1)** is the strictest Basel III measure of a bank's core capital strength relative to risk-weighted assets — its ultimate shock absorber. By parsing FR Y-9C filings, the engine builds a directed interbank lending graph and computes **PageRank** and **Betweenness Centrality** to identify systemic anchors and hidden bottlenecks in the global credit network. The latest run's top global hubs by PageRank were **IDCBY, ACGBY, CICHY, BACHY,** and **JPM**.

---

## ⚙️ Architecture & Data Engineering

- **Macroeconomic State Variables** — pulls historical VIX, high-yield credit spreads, and Treasury yield curves from the FRED API, and synthetically computes the TED spread as a proxy for interbank trust.
- **Regulatory Parser Engine** — downloads `.ZIP` archives directly from FFIEC, safely handles EOF/memory issues, and extracts targeted MDRM codes (e.g. `BHCA7206` for derivatives notional) to build baseline balance sheets.
- **Risk Calculation Tier** — `statsmodels` for conditional quantile regressions, `scikit-learn` for rolling matrix factorizations (PCA).
- **UI/Visualization** — rendered dynamically via `Streamlit` and `Plotly`.

---

## 📂 Repository Structure

```
global-banking-stress-monitor/
│
├── archive/                  # Historical EDA notebooks, prototypes, and old CSVs
│   ├── data/
│   └── scripts/
│
├── data/
│   ├── processed/            # Engine outputs (SRISK, CoVaR, Centrality CSVs)
│   └── raw/                  # IMMUTABLE: FFIEC ZIPs, PDFs, API raw pulls
│
├── src/                      # Production Python engine
│   ├── build_macro_factors.py
│   ├── parse_fry9c_bulk.py
│   ├── build_baseline_network.py
│   ├── network_analysis.py
│   ├── calculate_absorption.py
│   ├── risk_metrics.py
│   ├── calculate_srisk.py
│   └── dashboard.py
│
├── .env                      # API keys (FRED) — gitignored
├── requirements.txt          # Environment dependencies
└── run_pipeline.ps1          # Master orchestrator script
```

---

## 🚀 Getting Started

```powershell
# 1. Clone the repository
git clone https://github.com/Biswa1930/global-banking-stress-monitor.git
cd global-banking-stress-monitor

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your FRED API key to a .env file
#    FRED_API_KEY=your_key_here

# 5. Run the full pipeline (data ingestion → risk metrics → dashboard)
.\run_pipeline.ps1
```

Once the pipeline completes, the dashboard is served locally at:

```
http://localhost:8501
```

---

## 🛣️ Next Steps

- Add a fallback market-cap data source for tickers (e.g. BK) not resolved by the primary quote provider, to bring SRISK coverage to all 8 US G-SIBs.
- Extend the physics-informed deep learning (PINN) layer for forward-looking stress propagation, building on the point-kinetics systemic risk research this project draws from.
- Migrate remaining `use_container_width` Streamlit calls to the `width` parameter ahead of its deprecation.

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.