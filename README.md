# global-banking-stress-monitor
Real-time systemic risk early warning system using network topology, market microstructure, and physics-informed deep learning on 50 G-SIBs.


## 📊 Data Architecture & Methodology

This systemic risk engine operates on a hybrid data pipeline, combining automated high-frequency market data with a proprietary, manually engineered interbank network dataset. 

### 1. Data Retrieval & Engineering
Standard financial APIs do not capture the granular counterparty risk required for true systemic network topology. To build the foundational `network_edges.csv` matrix, data was sourced via a dual approach:
* **Manual Regulatory Extraction (The Network Proxy):** Wholesale credit exposure to financial institutions was manually extracted from the "Credit Risk Concentrations" and "Corporate Credit Portfolio" footnotes of the 2023 SEC 10-K Annual Reports for the major G-SIBs. 
* **Regulatory Balance Sheets:** Tier 1 capital ratios, total assets, and non-performing loans were pulled directly from FR Y-9C filings via the Chicago Fed public database.
* **Automated Market Feeds:** Daily equity microstructure data, macroeconomic stress indicators (VIX, TED spread, yield curve slopes), and CDS spreads were aggregated programmatically via `yfinance`, FRED, and OpenBB.

### 2. Mathematical Framework & Risk Metrics
The engine fuses traditional econometric risk models with advanced network topology and deep learning. The core calculations driving the dashboard include:

* **$\Delta\text{CoVaR}$ (Adrian & Brunnermeier):** Measures a specific bank's marginal contribution to overall systemic risk using quantile regression.
    $\Delta\text{CoVaR}_i = \text{CoVaR}(\text{system} | \text{bank}_i \text{ in distress}) - \text{CoVaR}(\text{system} | \text{bank}_i \text{ at median})$

* **SRISK (Brownlees & Engle):** Calculates the expected capital shortfall of a financial entity during a severe market decline. 
    $\text{SRISK}_i = E_i \times (k - (1-k) \times \text{LRMES}_i)$
    *(Where $E_i$ is equity, $k$ is the prudential capital ratio (8%), and $\text{LRMES}_i$ is the Long-Run Marginal Expected Shortfall).*

* **Network Centrality (Topology Layer):**
    Utilizes NetworkX to compute Eigenvector and Betweenness centrality on the directed weighted graph of interbank exposures, identifying critical hub nodes and transmission vectors prior to distress events.

* **Physics-Informed Deep Learning (PIDL):**
    Adapts Fokker-Planck dynamics and Ruppeiner curvature via a PyTorch Autoencoder to detect thermodynamic phase transitions in the global banking network. Spikes in the approximated scalar curvature act as an early warning signal for systemic instability:
    $R \sim -1 / \sqrt{\det(g)}$