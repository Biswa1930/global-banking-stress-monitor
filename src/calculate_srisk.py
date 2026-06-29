import pandas as pd
import numpy as np
import os

print("📉 Initializing SRISK Capital Shortfall Engine...")

# --- DYNAMIC PATH HANDLING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_dir, 'data')):
    data_dir = os.path.join(script_dir, 'data') # Script in root
else:
    data_dir = os.path.join(script_dir, '..', 'data') # Script in src/

try:
    df_market = pd.read_csv(os.path.join(data_dir, 'daily_market_factors.csv'), index_col=0)
    df_nodes = pd.read_csv(os.path.join(data_dir, 'network_nodes.csv'))
except FileNotFoundError:
    print("❌ Error: Could not find data files. Make sure you ran fetch_market_data.py!")
    exit()

bank_tickers = df_nodes['bank_id'].tolist()
return_cols = [f"{b}_Return" for b in bank_tickers if f"{b}_Return" in df_market.columns]

# 1. Define the Global Market Return
df_market['Global_System_Return'] = df_market[return_cols].mean(axis=1)

# 2. Identify Crisis Days (The worst 5% of days in the last 5 years)
threshold = df_market['Global_System_Return'].quantile(0.05)
crisis_days = df_market[df_market['Global_System_Return'] <= threshold]

print(f"🔬 Analyzing {len(crisis_days)} extreme market stress events...")

srisk_results = []
PRUDENTIAL_RATIO = 0.08  # The 8% 'k' value from your tracker

# 3. Calculate SRISK for each bank
for bank in bank_tickers:
    col_name = f"{bank}_Return"
    
    if col_name not in df_market.columns:
        continue # Skip if market data isn't fetched yet
        
    # A. Calculate MES (Marginal Expected Shortfall)
    # Average return of the bank during global crisis days
    mes = crisis_days[col_name].mean()
    
    # B. Calculate LRMES (Long-Run Marginal Expected Shortfall)
    # NYU V-Lab approximation for a 6-month crisis window
    lrmes = 1 - np.exp(18 * mes)
    
    # C. Extract Balance Sheet Fundamentals (Convert to Billions)
    assets = df_nodes.loc[df_nodes['bank_id'] == bank, 'total_assets_bln'].values[0]
    equity = df_nodes.loc[df_nodes['bank_id'] == bank, 'total_equity_bln'].values[0]
    debt = assets - equity
    
    # D. Calculate SRISK (Capital Shortfall in $ Billions)
    # Formula: k * Debt - (1 - k) * Equity * (1 - LRMES)
    srisk = (PRUDENTIAL_RATIO * debt) - ((1 - PRUDENTIAL_RATIO) * equity * (1 - lrmes))
    
    # E. Cap at 0 (If SRISK is negative, the bank has a surplus, not a shortfall)
    srisk_value = max(0, srisk)
    
    srisk_results.append({
        'Bank': bank,
        'MES (Daily %)': round(mes * 100, 2),
        'LRMES (6-Month Loss %)': round(lrmes * 100, 1),
        'Total Assets ($B)': assets,
        'SRISK Shortfall ($B)': round(srisk_value, 2)
    })

# 4. Format and Display
df_srisk = pd.DataFrame(srisk_results)
df_srisk = df_srisk.sort_values(by='SRISK Shortfall ($B)', ascending=False).reset_index(drop=True)

print("\n" + "="*85)
print(" 🚨 GLOBAL SYSTEMIC CAPITAL SHORTFALL (SRISK RANKINGS) 🚨")
print("="*85)
print(df_srisk.to_string(index=False))
print("="*85)