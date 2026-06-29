import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

print("🌀 Initializing Systemic Absorption Ratio (PCA) Engine...")

# --- BULLETPROOF PATH HANDLING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_dir, 'data')):
    data_dir = os.path.join(script_dir, 'data') 
else:
    data_dir = os.path.join(script_dir, '..', 'data') 

# Load the market factors we just updated
market_path = os.path.join(data_dir, 'daily_market_factors.csv')
try:
    df_market = pd.read_csv(market_path, index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ Error: Could not find daily_market_factors.csv. Run fetch_market_data.py first!")
    exit()

# Extract only the bank return columns
bank_cols = [col for col in df_market.columns if '_Return' in col]
df_returns = df_market[bank_cols].dropna()

print(f"📊 Analyzing cross-sectional variance for {len(bank_cols)} Global G-SIBs...")

# We use a 1-year rolling window (approx 252 trading days)
window_size = 252 
absorption_ratios = []
dates = []

print("🧮 Running rolling Principal Component Analysis (PCA)...")
for i in range(window_size, len(df_returns)):
    window_data = df_returns.iloc[i-window_size:i]
    
    # 1st Principal Component = The "Systemic" Factor
    pca = PCA(n_components=1)
    pca.fit(window_data)
    
    # Absorption Ratio: Fraction of variance explained by PC1
    ar = pca.explained_variance_ratio_[0]
    
    absorption_ratios.append(ar)
    dates.append(df_returns.index[i])

# Create time-series series
ar_series = pd.Series(absorption_ratios, index=dates)

# --- PLOTTING ---
# 1. Force the data to be numeric, converting errors to NaN
ar_series = pd.to_numeric(ar_series, errors='coerce')

# 2. Drop the NaNs so Matplotlib doesn't crash
ar_clean = ar_series.dropna()

plt.figure(figsize=(12, 6))
# 3. Use the cleaned data for plotting
plt.plot(ar_clean.index, ar_clean.values, color='#dc2626', linewidth=2, label='Absorption Ratio (1st PC)')
plt.axhline(y=0.70, color='black', linestyle='--', label='Contagion Threshold (70%)')

# 4. Fill between, only using the valid data
plt.fill_between(
    ar_clean.index, 
    ar_clean.values, 
    0.70, 
    where=(ar_clean.values > 0.70), 
    color='red', 
    alpha=0.3
)

plt.title("Systemic Absorption Ratio: Global Banking Contagion Index", fontsize=16, fontweight='bold')
plt.ylabel("Fraction of Variance Explained", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the chart
output_path = os.path.join(script_dir, '..', "absorption_ratio_chart.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Saved Absorption Ratio time-series chart to: {output_path}")

# Save the chart
output_path = os.path.join(script_dir, '..', "absorption_ratio_chart.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Saved Absorption Ratio time-series chart to: {output_path}")

# --- CURRENT STATE OUTPUT ---
current_ar = ar_series.iloc[-1]
print("\n" + "="*70)
print(f" 🚨 CURRENT MARKET STATE (As of {ar_series.index[-1].strftime('%Y-%m-%d')}) 🚨")
print("="*70)
print(f"Current Absorption Ratio: {current_ar*100:.2f}%")
if current_ar > 0.70:
    print("⚠️ WARNING: The system is highly coupled. Contagion risk is SEVERE.")
else:
    print("✅ CLEAR: Global banks are trading relatively independently.")
print("="*70)