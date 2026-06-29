import pandas as pd
import yfinance as yf
import numpy as np
import os

print("📡 Initiating Robust Market Data Fetch Pipeline...")

# --- BULLETPROOF PATH HANDLING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_dir, 'data')):
    data_dir = os.path.join(script_dir, 'data') 
else:
    data_dir = os.path.join(script_dir, '..', 'data') 

nodes_path = os.path.join(data_dir, 'network_nodes.csv')
output_path = os.path.join(data_dir, 'daily_market_factors.csv')

# 1. Load bank tickers
try:
    df_nodes = pd.read_csv(nodes_path)
    tickers = df_nodes['bank_id'].tolist()
except FileNotFoundError:
    print(f"❌ Error: Could not find {nodes_path}")
    exit()

state_variables = ['^VIX', '^IRX', '^TNX']
all_tickers = tickers + state_variables

print(f"📥 Downloading 5-year historical data for {len(all_tickers)} assets...")

# 2. Download daily closing prices
df_raw = yf.download(all_tickers, start="2019-01-01", end="2024-01-01")

if 'Close' in df_raw.columns.get_level_values(0):
    raw_data = df_raw['Close']
else:
    raw_data = df_raw.xs('Close', level=1, axis=1)

# --- THE FIX: Forward-Fill and Backward-Fill missing holiday prices ---
raw_data = raw_data.ffill().bfill()

# 3. Compute continuous log returns
df_returns = pd.DataFrame()
for bank in tickers:
    df_returns[f'{bank}_Return'] = np.log(raw_data[bank] / raw_data[bank].shift(1))

df_returns['VIX_Level'] = raw_data['^VIX']
df_returns['Yield_Curve_Slope'] = raw_data['^TNX'] - raw_data['^IRX']

# Drop ONLY the very first row (which is naturally NaN due to the shift calculation)
df_returns = df_returns.iloc[1:]

# 4. Export to the correct data/ folder
df_returns.to_csv(output_path)
print(f"✅ Saved 'daily_market_factors.csv' to: {output_path}")
print(f"📊 Dataset successfully retained {len(df_returns)} trading days.")