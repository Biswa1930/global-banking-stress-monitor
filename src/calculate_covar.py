import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import yfinance as yf
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore') # Hides unnecessary statistical warnings

print("🧮 Initializing LIVE Delta-CoVaR Systemic Risk Engine...")

# 1. Load the static network structure (Nodes and Edges)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

try:
    df_nodes = pd.read_csv(os.path.join(data_dir, 'network_nodes.csv'))
    df_edges = pd.read_csv(os.path.join(data_dir, 'network_edges.csv'))
except FileNotFoundError:
    df_nodes = pd.read_csv('network_nodes.csv')
    df_edges = pd.read_csv('network_edges.csv')

bank_tickers = df_nodes['bank_id'].tolist()
state_variables = ['^VIX', '^IRX', '^TNX']
all_tickers = bank_tickers + state_variables

# 2. Fetch a rolling 5-year window of data up to TODAY
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365) # 5 years ago

print(f"📡 Downloading 5-year rolling market data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
df_raw = yf.download(all_tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)

# Safely extract 'Close' prices
if 'Close' in df_raw.columns.get_level_values(0):
    raw_data = df_raw['Close']
else:
    raw_data = df_raw.xs('Close', level=1, axis=1)

# 3. Process the Live Data into Log Returns and Macro Factors
df_market = pd.DataFrame()
for bank in bank_tickers:
    df_market[f'{bank}_Return'] = np.log(raw_data[bank] / raw_data[bank].shift(1))

df_market['VIX_Level'] = raw_data['^VIX']
df_market['Yield_Curve_Slope'] = raw_data['^TNX'] - raw_data['^IRX']
df_market.dropna(inplace=True)

# 4. Create the System Proxy Index
return_cols = [f"{b}_Return" for b in bank_tickers]
df_market['System_Return'] = df_market[return_cols].mean(axis=1)

print("🔬 Running Live Quantile Regressions for all US G-SIBs...")

results = []

# 5. Calculate Delta CoVaR
for bank in bank_tickers:
    bank_ret = f"{bank}_Return"
    
    # Formula: System Return ~ Bank Return + Macro Factors
    formula = f"System_Return ~ {bank_ret} + VIX_Level + Yield_Curve_Slope"
    
    # Run the regressions
    mod_median = smf.quantreg(formula, df_market).fit(q=0.5)
    mod_distress = smf.quantreg(formula, df_market).fit(q=0.05)
    
    delta_covar = abs(mod_distress.params[bank_ret] - mod_median.params[bank_ret])
    
    # Fetch exposure and size from your proprietary dataset
    assets = df_nodes.loc[df_nodes['bank_id'] == bank, 'total_assets_bln'].values[0]
    exposure = df_edges.loc[df_edges['lender'] == bank, 'exposure_bln'].values[0]
    
    # Composite Risk Score
    systemic_score = delta_covar * (assets / 1000) * (exposure / 100)
    
    results.append({
        'Bank': bank,
        'Delta_CoVaR (Spillover %)': round(delta_covar, 4),
        'Total Assets ($B)': assets,
        'Interbank Exposure ($B)': exposure,
        'Systemic Risk Score': round(systemic_score, 2)
    })

# 6. Format and Display Results
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='Systemic Risk Score', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print(f" 🚨 LIVE SYSTEMIC RISK RANKINGS (As of {end_date.strftime('%Y-%m-%d')}) 🚨")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)