import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings
warnings.filterwarnings('ignore')

print("🏦 Initializing Phase 2C: SRISK Engine (Brownlees & Engle)...")

# 1. Define Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
returns_file = os.path.join(processed_dir, 'bank_log_returns.csv')
reg_file = os.path.join(processed_dir, 'us_banks_regulatory_raw.csv')
output_file = os.path.join(processed_dir, 'srisk_results.csv')

# 2. Load the Data
df_returns = pd.read_csv(returns_file, parse_dates=['Date']).set_index('Date')
df_reg = pd.read_csv(reg_file)

# We need the most recent total assets (2023)
df_2023 = df_reg[df_reg['Year'] == 2023]

# Create a proxy for the Market Return (Equally weighted system return)
bank_cols = df_returns.columns
df_returns['Market'] = df_returns.mean(axis=1)

print("📉 Calculating Marginal Expected Shortfall (MES)...")
# Define tail events: The worst 5% of market days
q5 = df_returns['Market'].quantile(0.05)
crisis_days = df_returns[df_returns['Market'] <= q5]

srisk_data = []
k = 0.08  # 8% prudential capital requirement

for bank in bank_cols:
    try:
        # 1. Calculate MES (Average daily return on market crisis days)
        mes = crisis_days[bank].mean()
        
        # 2. Approximate LRMES for a 6-month horizon
        lrmes = 1 - np.exp(18 * mes)
        
        # 3. Get Total Assets from the regulatory tape (in Billions)
        assets_b = df_2023.loc[df_2023['Bank_Ticker'] == bank, 'Total_Assets_Billion'].values[0]
        
        # 4. Get Current Market Equity via yfinance (in Billions)
        ticker = yf.Ticker(bank)
        equity_b = ticker.info.get('marketCap', 0) / 1_000_000_000
        
        if equity_b == 0:
            print(f"⚠️ Warning: Could not fetch Market Cap for {bank}. Skipping...")
            continue
            
        # Book value of Debt = Total Assets - Market Equity
        debt_b = assets_b - equity_b 
        
        # 5. Calculate SRISK
        srisk = (k * debt_b) - ((1 - k) * equity_b * (1 - lrmes))
        
        # Aggregate SRISK truncates negative values (surpluses) to 0
        srisk_truncated = max(0, srisk)
        
        srisk_data.append({
            'Bank_Ticker': bank,
            'MES_Daily': round(mes, 4),
            'LRMES_6Mo': round(lrmes, 4),
            'Equity_Billion': round(equity_b, 2),
            'Debt_Billion': round(debt_b, 2),
            'SRISK_Billion': round(srisk_truncated, 2)
        })
        
    except Exception as e:
        print(f"⚠️ Error processing {bank}: {e}")

# 6. Compile and Save
df_srisk = pd.DataFrame(srisk_data).sort_values(by='SRISK_Billion', ascending=False)
df_srisk.to_csv(output_file, index=False)

print("\n" + "="*80)
print(f"✅ SUCCESSFULLY CALCULATED SRISK FOR {len(df_srisk)} BANKS")
print("="*80)
print(f"Data saved to: {output_file}")
print("\n🚨 Top 5 Systemically Risky Banks by Capital Shortfall (SRISK):")
print(df_srisk[['Bank_Ticker', 'LRMES_6Mo', 'Equity_Billion', 'Debt_Billion', 'SRISK_Billion']].head(5).to_string(index=False))