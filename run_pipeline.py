"""
Global Banking Stress Monitor - Core Orchestrator
-------------------------------------------------
Executes the data ingestion pipeline and passes engineered assets
into the network topology and market microstructure risk engines.
"""

import os
import pandas as pd
import numpy as np
from src.data_pipeline import fetch_macro_fred, fetch_bank_microstructure, build_coverage_report
from src.network_analysis import build_interbank_network, compute_centrality_metrics
from src.risk_metrics import compute_delta_covar, compute_srisk
from dotenv import load_dotenv 

load_dotenv()
def main():
    print("==================================================")
    print("🚀 STARTING GLOBAL BANKING STRESS MONITOR RUNNER")
    print("==================================================\n")

    # ----------------------------------------------------------------
    # PHASE 1: DATA INGESTION
    # ----------------------------------------------------------------
    print("--- PHASE 1: Executing Data Ingestion Pipeline ---")
    tickers = ["JPM", "BAC", "C", "WFC", "GS", "MS", "BK", "STT"]
    start_date = "2020-01-01"
    end_date = "2026-06-01"
    
    df_returns, df_prices = fetch_bank_microstructure(tickers, start_date, end_date)
    # Grab the API key securely from the environment
    fred_key = os.getenv("FRED_API_KEY")
    macro_df = fetch_macro_fred(api_key=fred_key, start_date=start_date, end_date=end_date)
    
    if not df_prices.empty:
        report_df = build_coverage_report(df_prices, macro_df)
        os.makedirs("data/processed", exist_ok=True)
        df_returns.to_csv("data/processed/bank_log_returns.csv")
        df_prices.to_csv("data/processed/bank_raw_prices.csv")
        print("💾 Phase 1 assets successfully cached.")
    else:
        print("❌ Phase 1 Data Ingestion failed. Aborting.")
        return

    print("\n--------------------------------------------------")
    # ----------------------------------------------------------------
    # PHASE 2A: NETWORK TOPOLOGY ANALYSIS
    # ----------------------------------------------------------------
    print("--- PHASE 2A: Executing Network Topology Engine ---")
    edge_file_path = "data/processed/network_edges.csv"
    
    if os.path.exists(edge_file_path):
        edge_df = pd.read_csv(edge_file_path)
        interbank_graph = build_interbank_network(edge_df)
        centrality_df = compute_centrality_metrics(interbank_graph)
        centrality_df.to_csv("data/processed/network_centrality_metrics.csv")
    else:
        print(f"⚠️ Missing '{edge_file_path}'. Run 'python generate_mock_network.py' first.")
        return

    print("\n--------------------------------------------------")
    # ----------------------------------------------------------------
    # PHASE 2B: MARKET MICROSTRUCTURE RISK METRICS
    # ----------------------------------------------------------------
    print("--- PHASE 2B: Executing Econometric Risk Engine ---")
    
    # Create an equally-weighted systemic index from our G-SIBs to act as the 'system' proxy
    df_returns['SYSTEM_INDEX'] = df_returns[tickers].mean(axis=1)
    
    risk_records = []
    
    # Proxy liabilities values in Billions USD for SRISK calculation 
    liability_book = {
        "JPM": 3400.0, "BAC": 2900.0, "C": 2200.0, "WFC": 1700.0,
        "GS": 1500.0,  "MS": 1100.0,  "BK": 400.0,  "STT": 300.0
    }

    print("📊 Estimating Delta CoVaR via Quantile Regression & SRISK Shortfalls...")
    for bank in tickers:
        d_covar = compute_delta_covar(df_returns, target_bank=bank, system_index='SYSTEM_INDEX', q=0.05)
        equity_proxy = 200.0  # Proxy market cap base in Billions
        debt_proxy = liability_book.get(bank, 500.0)
        srisk_val = compute_srisk(equity_val=equity_proxy, debt_val=debt_proxy, lrmes=0.40, k=0.08)
        
        risk_records.append({
            "Bank": bank,
            "Delta_CoVaR": round(d_covar, 6) if not np.isnan(d_covar) else 0.0,
            "SRISK_Shortfall_Billions": round(srisk_val, 2)
        })
        
    df_risk = pd.DataFrame(risk_records).set_index("Bank")
    df_risk.to_csv("data/processed/microstructure_risk_metrics.csv")
    
    # Merge topology with econometric risk metrics
    consolidated_df = centrality_df.join(df_risk)
    consolidated_df.to_csv("data/processed/consolidated_systemic_risk_matrix.csv")
    
    print("\n🏆 Consolidated Macro-Prudential Risk Matrix (Top Entities):")
    print(consolidated_df[['Eigenvector_Systemic_Importance', 'Delta_CoVaR', 'SRISK_Shortfall_Billions']].head(4))

    print("\n==================================================")
    print("✅ RUNNER EXECUTION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()