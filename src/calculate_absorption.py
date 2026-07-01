import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("🌀 Initializing Phase 2B: Systemic Absorption Ratio (PCA) Engine...")

# 1. Define Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
market_data_file = os.path.join(processed_dir, 'daily_market_factors.csv')
output_csv = os.path.join(processed_dir, 'absorption_ratio.csv')

# 2. Load the Data
df = pd.read_csv(market_data_file, parse_dates=['Date'])
df = df.set_index('Date')

# 3. Isolate the 8 US G-SIBs (Drop the macro variables for the PCA)
bank_cols = [col for col in df.columns if col not in ['VIX', 'HY_CREDIT', 'YIELD_CURVE', 'TED_SPREAD']]
df_banks = df[bank_cols]

print(f"📊 Analyzing cross-sectional variance for {len(bank_cols)} US G-SIBs...")
print("🧮 Running rolling Principal Component Analysis (PCA)...")

# 4. Rolling PCA Setup
window_size = 252 # 1 Trading Year
absorption_ratios = []
dates = []

# Iterate through the time series with a rolling window
for i in range(window_size, len(df_banks)):
    # Extract the 1-year window of log returns
    window_data = df_banks.iloc[i - window_size : i]
    
    # Standardize the data (PCA requires mean=0, variance=1)
    standardized_data = (window_data - window_data.mean()) / window_data.std()
    
    # Fit PCA
    pca = PCA(n_components=1) # We only care about the 1st Principal Component (Systemic Risk)
    pca.fit(standardized_data)
    
    # Extract the variance explained by the 1st PC
    ar = pca.explained_variance_ratio_[0]
    
    absorption_ratios.append(ar)
    dates.append(df_banks.index[i])

# 5. Compile Results
df_ar = pd.DataFrame({'Date': dates, 'Absorption_Ratio': absorption_ratios})

if not df_ar.empty:
    df_ar.to_csv(output_csv, index=False)
    print("\n" + "="*80)
    print(f"✅ SUCCESSFULLY CALCULATED ABSORPTION RATIO ({len(df_ar)} Rolling Windows)")
    print("="*80)
    print(f"Data saved to: {output_csv}")
    
    # Check the highest fragility days
    highest_stress = df_ar.sort_values(by='Absorption_Ratio', ascending=False).head(5)
    print("\n🚨 Top 5 Days of Maximum Systemic Fragility (High Correlation):")
    print(highest_stress.to_string(index=False))
else:
    print("❌ Failed to calculate Absorption Ratio. Check data length.")