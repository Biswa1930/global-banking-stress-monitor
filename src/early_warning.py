import pandas as pd
import yfinance as yf
import os

print("🚨 Initializing REAL-TIME Early Warning System...")

# 1. Load your Bank Tickers from the database
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

try:
    df_nodes = pd.read_csv(os.path.join(data_dir, 'network_nodes.csv'))
except FileNotFoundError:
    df_nodes = pd.read_csv('network_nodes.csv')

bank_tickers = df_nodes['bank_id'].tolist()
state_variables = ['^VIX', '^IRX', '^TNX']
all_tickers = bank_tickers + state_variables

# 2. Fetch LIVE Data (We pull 5 days to ensure we catch the latest valid trading day)
print("📡 Fetching live market data from Wall Street...")
df_raw = yf.download(all_tickers, period="5d", progress=False)

# Safely extract 'Close' prices handling yfinance version differences
if 'Close' in df_raw.columns.get_level_values(0):
    raw_data = df_raw['Close']
else:
    raw_data = df_raw.xs('Close', level=1, axis=1)

# 3. Isolate Today and Yesterday for live comparisons
today_data = raw_data.iloc[-1]
yesterday_data = raw_data.iloc[-2]

# Format the date so we know exactly what we are looking at
latest_date = raw_data.index[-1].strftime('%Y-%m-%d')

print(f"📅 Live Scan Date: {latest_date} (Market Close)\n")

alerts = []

# --- TRIPWIRE 1: Systemic Fear (VIX) ---
vix_today = today_data['^VIX']
if vix_today > 30:
    alerts.append(f"⚠️ MACRO ALERT: VIX is dangerously elevated at {vix_today:.2f} (Threshold: 30)")

# --- TRIPWIRE 2: Recession Indicator (Yield Curve Inversion) ---
yield_curve = today_data['^TNX'] - today_data['^IRX']
if yield_curve < 0:
    alerts.append(f"⚠️ MACRO ALERT: Yield Curve is inverted at {yield_curve:.2f}%")

# --- TRIPWIRE 3: G-SIB Equity Contagion ---
# Calculate real-time 1-day percentage drops for all 8 banks
for bank in bank_tickers:
    price_today = today_data[bank]
    price_yesterday = yesterday_data[bank]
    pct_change = ((price_today - price_yesterday) / price_yesterday) * 100
    
    if pct_change <= -5.0:
        alerts.append(f"🩸 BANK ALERT: {bank} equity dropped by {pct_change:.2f}% today.")

# 4. Report the Findings
if not alerts:
    print("✅ ALL CLEAR: No systemic tripwires triggered today.")
    print("Status: Normal Market Conditions.")
else:
    print("!"*60)
    print(" 🔴 SYSTEMIC WARNINGS DETECTED 🔴")
    print("!"*60)
    for alert in alerts:
        print(alert)
    print("!"*60)
    print("\nRecommendation: Run calculate_covar.py immediately to assess spillover risk.")