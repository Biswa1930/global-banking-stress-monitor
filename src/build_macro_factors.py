import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

print("🌍 Initializing Phase 1B Supplement: Macro Factor Engine...")

# 1. Setup Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
returns_file = os.path.join(processed_dir, 'bank_log_returns.csv')
output_file = os.path.join(processed_dir, 'daily_market_factors.csv')

# 2. Load API Key
load_dotenv(os.path.join(script_dir, '..', '.env'))
fred_api_key = os.getenv('FRED_API_KEY')

if not fred_api_key:
    raise ValueError("❌ FRED_API_KEY not found in .env file. Please ensure your .env exists in the root folder.")

fred = Fred(api_key=fred_api_key)

# 3. Load Bank Returns (This acts as our master trading calendar)
df_banks = pd.read_csv(returns_file, parse_dates=['Date'])
df_banks = df_banks.set_index('Date')

# Date limits based on your tracker
start_date = '2019-01-01'
end_date = '2024-01-01'

print("📥 Downloading official state variables from Federal Reserve (FRED)...")
try:
    vix = fred.get_series('VIXCLS', observation_start=start_date, observation_end=end_date)
    hy_spread = fred.get_series('BAMLH0A0HYM2', observation_start=start_date, observation_end=end_date)
    yield_curve = fred.get_series('T10Y2Y', observation_start=start_date, observation_end=end_date)
    
    # TED Spread components
    sofr = fred.get_series('SOFR', observation_start=start_date, observation_end=end_date)
    dtb3 = fred.get_series('DTB3', observation_start=start_date, observation_end=end_date)
except Exception as e:
    raise ConnectionError(f"❌ Failed to download from FRED: {e}")

# 4. Consolidate Macro Data
df_macro = pd.DataFrame({
    'VIX': vix,
    'HY_CREDIT': hy_spread,
    'YIELD_CURVE': yield_curve,
    'SOFR': sofr,
    'DTB3': dtb3
})

# Calculate manual TED spread as per your research notes
df_macro['TED_SPREAD'] = df_macro['SOFR'] - df_macro['DTB3']

# Drop the raw rate components to prevent multicollinearity in the regression
df_macro = df_macro.drop(columns=['SOFR', 'DTB3'])

print("🔗 Merging macro state variables with bank log returns...")

# 5. Left Join onto Bank Calendar and handle misalignment
# A left join ensures we only keep days where the stock market was actually open.
df_final = df_banks.join(df_macro, how='left')

# Forward-fill macro data for days when stocks traded but the bond market/FRED was closed
df_final = df_final.ffill().dropna()

df_final.reset_index(inplace=True)
df_final.to_csv(output_file, index=False)

print("\n" + "="*80)
print(f"✅ SUCCESSFULLY BUILT MARKET FACTORS ({len(df_final)} Trading Days)")
print("="*80)
print(f"Data saved to: {output_file}")
print("\n🔍 Verification Check (First 5 Days of Macro Integration):")
print(df_final[['Date', 'JPM', 'VIX', 'HY_CREDIT', 'YIELD_CURVE', 'TED_SPREAD']].head().to_string(index=False))