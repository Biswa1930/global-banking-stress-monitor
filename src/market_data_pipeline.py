import pandas as pd
import yfinance as yf
import numpy as np
import os
from pathlib import Path
import warnings

# Suppress yfinance warnings for clean terminal output
warnings.filterwarnings('ignore')

def fetch_market_returns(nodes_file_path, output_dir):
    print("📊 Initializing Phase 1B: Market Data Pipeline (yfinance)...")
    
    # 1. Load the validated tickers from Phase 1A
    nodes_path = Path(nodes_file_path)
    if not nodes_path.exists():
        raise FileNotFoundError(f"System Error: Run data_ingestion.py first. Missing {nodes_path.resolve()}")
        
    df_nodes = pd.read_csv(nodes_path)
    bank_tickers = df_nodes['Bank_Ticker'].dropna().unique().tolist()
    
    # Add S&P 500 as the proxy for the global financial system
    system_ticker = "^GSPC" 
    download_list = bank_tickers + [system_ticker]
    
    print(f"📡 Fetching 5-year daily pricing for {len(bank_tickers)} banks + System Index...")
    
    # 2. Bulk download via yfinance (5 years of daily data)
    # yfinance auto-threads this, making it extremely fast
    raw_data = yf.download(download_list, period="5y", interval="1d", auto_adjust=True)['Close']
    
    # 3. Compute Log Returns
    print("🧮 Computing daily log returns...")
    # Log returns are standard for quantitative risk modeling (CoVaR/GARCH)
    daily_returns = np.log(raw_data / raw_data.shift(1)).dropna(how='all')
    
    # Rename the system index for downstream compatibility with risk_metrics.py
    daily_returns.rename(columns={system_ticker: 'SYSTEM_RETURN'}, inplace=True)
    
    # 4. Data Hygiene: Drop empty columns (delisted ADRs or invalid yfinance tickers)
    daily_returns = daily_returns.dropna(axis=1, how='all')
    
    # Ensure date is a proper column, not just an index, for CSV saving
    daily_returns.reset_index(inplace=True)
    
    # 5. Pipeline Handoff
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "daily_returns.csv"
    
    daily_returns.to_csv(output_path, index=False)
    
    # 6. Audit Logging
    print("-" * 50)
    print(f"Trading Days Captured:  {len(daily_returns)}")
    print(f"Valid Tickers Fetched:  {len(daily_returns.columns) - 2}") # Subtracting Date and SYSTEM_RETURN
    print("-" * 50)
    print(f"Phase 1B Complete: Market data saved to {output_path.resolve()}\n")

# --- Execution ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Inputs and Outputs mapping directly to your GitHub structure
    INPUT_FILE = os.path.join(script_dir, '..', 'data', 'processed', 'cleaned_bank_nodes.csv')
    OUTPUT_DIR = os.path.join(script_dir, '..', 'data', 'processed')
    
    fetch_market_returns(INPUT_FILE, OUTPUT_DIR)