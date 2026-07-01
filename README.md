# 🏦 Global Banking Stress Monitor (GBSM)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Quant Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-005660)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 📌 Executive Summary
The **Global Banking Stress Monitor** is an institutional-grade, end-to-end quantitative pipeline designed to measure, model, and visualize systemic risk across Global Systemically Important Banks (G-SIBs). 

Rather than relying on isolated metrics like standard volatility, this engine aggregates four distinct dimensions of financial fragility: **Topological Interconnectedness, Systemic Spillovers, Market Fragility, and Capital Vulnerability.** It programmatically ingests messy regulatory filings (FFIEC FR Y-9C) and live market data to generate a unified risk dashboard.

---

## 🧠 The Mathematics of Systemic Risk (A Conceptual Guide)

Financial crises do not happen because a single bank makes a bad trade; they happen because banks are heavily interconnected. This project utilizes advanced financial mathematics to measure exactly how a shock to one institution ripples through the global economy. 

Below is an explanation of the core quantitative models running inside this engine.

### 1. $\Delta$CoVaR (Systemic Spillover Risk)
* **The Concept:** Traditional risk metrics like Value-at-Risk (VaR) measure how much a single bank might lose in a worst-case scenario. It treats banks like isolated islands. **CoVaR** (Conditional Value-at-Risk) measures how much the *entire financial system* stands to lose when one specific bank crashes. 
* **The Math:** Developed by Adrian & Brunnermeier (2016), $\Delta$CoVaR is the difference between the system's risk when a bank is operating normally versus when it is in distress.
* **In Plain English:** If VaR measures the risk of a single house burning down, $\Delta$CoVaR measures the risk that the fire will spread and burn down the entire neighborhood. We use 5th-percentile **Quantile Regression** conditioned on macroeconomic factors (VIX, Yield Curve) to calculate this.

### 2. SRISK (Capital Vulnerability)
* **The Concept:** While $\Delta$CoVaR looks outward (how much damage a bank *causes*), **SRISK** looks inward (how much damage a bank *takes*). 
* **The Math:** Developed by Brownlees & Engle (2017) at NYU, SRISK calculates a bank's expected capital shortfall if the global stock market were to collapse by 40% over six months. 

$$SRISK_i = k \cdot Debt_i - (1 - k) \cdot Equity_i \cdot (1 - LRMES_i)$$

* **In Plain English:** It represents the "Bailout Bill." If a massive recession hits, highly leveraged banks will burn through their cash reserves instantly. SRISK measures exactly how many billions of dollars in taxpayer or central bank bailouts a specific bank would need to survive.

### 3. Systemic Absorption Ratio (Market Fragility)
* **The Concept:** During a financial panic, diversification disappears. Investors sell everything at once, causing all bank stocks to move in perfect, terrifying unison.
* **The Math:** The engine runs a 252-day rolling **Principal Component Analysis (PCA)** across the covariance matrix of bank stock returns. It calculates what percentage of market movement is driven by the 1st Principal Component (the "Systemic Factor").
* **In Plain English:** It acts as an early warning radar. When the Absorption Ratio spikes above 70%, it means idiosyncratic (individual) bank traits no longer matter. The market has become highly fragile, tightly coupled, and primed for a cascading crash.

### 4. CET1 & Regulatory Network Topology
* **CET1 (Common Equity Tier 1):** The strictest measurement of a bank's core financial strength, mandated by the Basel III international framework. It represents the highest-quality capital a bank holds (mostly common stock and retained earnings) compared to its risk-weighted assets. It is the bank's ultimate shock absorber.
* **PageRank & Betweenness Centrality:** By parsing millions of rows of Federal Reserve FR Y-9C filings, this engine builds a directed graph of interbank lending. We use Google's original PageRank algorithm to identify which banks are the "systemic anchors" (borrowing from/lending to the most connected peers) and Betweenness Centrality to find the hidden bottlenecks in the global flow of credit.

---

## ⚙️ Architecture & Data Engineering

This project is fully automated and circumvents the need for manual data entry or excel formatting.

* **Macroeconomic State Variables:** Pings the Federal Reserve Economic Data (FRED) API to pull historical VIX, High-Yield Credit Spreads, and Treasury Yield Curves. Synthetically computes the TED spread to measure interbank trust.
* **Regulatory Parser Engine:** Custom bulk-processing script that downloads `.ZIP` archives directly from the FFIEC, safely bypasses EOF memory leaks, and extracts targeted MDRM accounting codes (e.g., `BHCA7206` for Derivatives Notional) to build baseline balance sheets.
* **Risk Calculation Tier:** Utilizes `statsmodels` for conditional quantile regressions and `scikit-learn` for rolling matrix factorizations.
* **UI/Visualization:** Renders the data structures dynamically via `Streamlit` and `Plotly`.

---

## 📂 Repository Structure

The repository enforces strict separation of concerns, ensuring production scripts are isolated from raw, immutable data.

```text
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
├── src/                      # Production Python Engine
│   ├── build_macro_factors.py
│   ├── parse_fry9c_bulk.py
│   ├── build_baseline_network.py
│   ├── network_analysis.py
│   ├── calculate_absorption.py
│   ├── risk_metrics.py
│   ├── calculate_srisk.py
│   └── dashboard.py
│
├── .env                      # API Keys (FRED) - GitIgnored
├── requirements.txt          # Environment dependencies
└── run_pipeline.ps1          # Master orchestrator script