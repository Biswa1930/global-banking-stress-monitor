import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import os
from dotenv import load_dotenv

print("📡 Initializing Phase 1B: Global Market & Macro Data Ingestion...")

# ==========================================
# 1. PATH & ENVIRONMENT HANDLING
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
env_path = os.path.join(script_dir, '..', '.env')

# Ensure output directory exists
os.makedirs(processed_dir, exist_ok=True)

# Load FRED API Key securely
load_dotenv(env_path)
fred_key = os.getenv('FRED_API_KEY')
if not fred_key:
    print("⚠️ Warning: FRED_API_KEY not found in .env file. Macro data will be skipped.")

# ==========================================
# 2. MARKET DATA: 39 G-SIB LOG RETURNS
# ==========================================
print("📈 Downloading 5-Year Equity Data for 39 G-SIBs...")

gsib_tickers = [
    'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'BK', 'STT',           # US
    'HSBC', 'BCS', 'DB', 'UBS', 'ING', 'SAN', 'BNPQY', 'SCGLY',  # EU (ADRs/Primary)
    'CRARY', 'SCBFF', 'UNCFF', 'NRDBY', 'DNKEY', 'ABN.AS',       # EU Continued
    'MUFG', 'SMFG', 'MFG',                                       # Japan
    'IDCBY', 'CICHY', 'ACGBY', 'BACHY',                          # China
    'DBSDY', 'HDB', 'CMWAY', 'WEBNF', 'ANZGY', 'NABZY',          # RoW / APAC
    'RY', 'TD', 'BNS', 'BMO', 'CM'                               # Canada
]

# Download data (2019-01-01 to 2024-01-01 to capture COVID & SVB crashes)
data = yf.download(gsib_tickers, start="2019-01-01", end="2024-01-01", auto_adjust=False)

# Extract Adjusted Close prices
adj_close = data['Adj Close']

# Calculate Daily Logarithmic Returns
print("🧮 Computing Logarithmic Returns...")
log_returns = np.log(adj_close / adj_close.shift(1)).dropna(how='all')

# Save to CSV
returns_path = os.path.join(processed_dir, 'gsib_log_returns.csv')
log_returns.to_csv(returns_path)
print(f"✅ Equity data saved to: {returns_path}")

# ==========================================
# 3. MACROECONOMIC STATE VARIABLES (FRED)
# ==========================================
if fred_key:
    print("🏦 Pinging Federal Reserve (FRED) API for Macro State Variables...")
    try:
        fred = Fred(api_key=fred_key)
        
        # Pull core indicators
        vix = fred.get_series('VIXCLS', observation_start='2019-01-01', observation_end='2024-01-01')
        hy_spread = fred.get_series('BAMLH0A0HYM2', observation_start='2019-01-01', observation_end='2024-01-01')
        yield_curve = fred.get_series('T10Y2Y', observation_start='2019-01-01', observation_end='2024-01-01')
        
        # Synthetic TED Spread (SOFR - 3M T-Bill) because TEDRATE was discontinued
        sofr = fred.get_series('SOFR', observation_start='2019-01-01', observation_end='2024-01-01')
        t_bill_3m = fred.get_series('DTB3', observation_start='2019-01-01', observation_end='2024-01-01')
        ted_spread_proxy = sofr - t_bill_3m
        
        # Compile into a single DataFrame
        macro_df = pd.DataFrame({
            'VIX': vix,
            'High_Yield_Spread': hy_spread,
            'Yield_Curve_10Y_2Y': yield_curve,
            'TED_Spread_Proxy': ted_spread_proxy
        }).ffill().dropna()
        
        # Save to CSV
        macro_path = os.path.join(processed_dir, 'macro_stress_indicators.csv')
        macro_df.to_csv(macro_path)
        print(f"✅ Macroeconomic data saved to: {macro_path}")
        
    except Exception as e:
        print(f"❌ Error fetching FRED data: {e}")

print("\n" + "="*80)
print("🚀 PHASE 1B COMPLETE. System is ready for Risk Calculations.")
print("="*80)