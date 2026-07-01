import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import warnings
warnings.filterwarnings('ignore') # Suppress statsmodels convergence warnings for extreme tails

print("📉 Initializing Phase 2B: Systemic Risk Engine (ΔCoVaR)...")

# 1. Define Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
market_data_file = os.path.join(processed_dir, 'daily_market_factors.csv')
output_file = os.path.join(processed_dir, 'covar_results.csv')

if not os.path.exists(market_data_file):
    raise FileNotFoundError(f"❌ Missing {market_data_file}. Please ensure Phase 1B output is correctly named.")

# 2. Load Market Data
df = pd.read_csv(market_data_file, parse_dates=['Date'])
df = df.sort_values('Date').set_index('Date')

# 3. Identify Bank Tickers vs Macro Indicators
# Assuming macro indicators have standard names like VIX, HY, TED, etc. 
# We treat all other columns as bank tickers.
macro_cols = [col for col in df.columns if any(m in col.upper() for m in ['VIX', 'HY', 'TED', 'YIELD', 'SOFR', 'DTB3'])]
bank_cols = [col for col in df.columns if col not in macro_cols]

print(f"📊 Found {len(bank_cols)} banks and {len(macro_cols)} macro state variables.")

# 4. Compute the "System Return" (Equally weighted banking index)
df['System_Return'] = df[bank_cols].mean(axis=1)

# 5. Lag the Macro State Variables (Critical to avoid Look-Ahead Bias)
for macro in macro_cols:
    df[f'{macro}_lag1'] = df[macro].shift(1)

# Drop the first row which now has NaNs due to the lag
df = df.dropna()

print("🧮 Running Adrian & Brunnermeier (2016) Quantile Regressions...")

results = []
q_distress = 0.05 # 5th percentile (Distress)
q_median = 0.50   # 50th percentile (Normal state)

# Build the regression formula for the state variables dynamically
state_var_formula = " + ".join([f"{m}_lag1" for m in macro_cols])

for bank in bank_cols:
    try:
        # Formula: System Return ~ Bank Return + Lagged Macro Variables
        formula = f"System_Return ~ {bank} + {state_var_formula}" if state_var_formula else f"System_Return ~ {bank}"
        
        # 1. Estimate Systemic Risk in Distress (5th Percentile)
        model_distress = smf.quantreg(formula, df).fit(q=q_distress, max_iter=2000)
        beta_distress = model_distress.params[bank]
        
        # 2. Estimate Systemic Risk in Normal State (Median)
        model_median = smf.quantreg(formula, df).fit(q=q_median, max_iter=2000)
        beta_median = model_median.params[bank]
        
        # 3. Calculate Bank's historical Value at Risk (VaR)
        var_distress = df[bank].quantile(q_distress)
        var_median = df[bank].quantile(q_median)
        
        # 4. Calculate ΔCoVaR
        # How much does the system's VaR widen when this specific bank moves from median to distress?
        delta_covar = beta_distress * (var_distress - var_median)
        
        results.append({
            'Bank_Ticker': bank,
            'VaR_5pct': round(var_distress, 4),
            'Beta_Distress': round(beta_distress, 4),
            'Delta_CoVaR': round(delta_covar, 6)
        })
    except Exception as e:
        print(f"⚠️ Could not compute ΔCoVaR for {bank}: {e}")

# 6. Compile and Save
df_covar = pd.DataFrame(results)

if not df_covar.empty:
    # Sort by the most systemically dangerous banks (most negative ΔCoVaR)
    df_covar = df_covar.sort_values(by='Delta_CoVaR', ascending=True)
    df_covar.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print(f"✅ SUCCESSFULLY CALCULATED ΔCoVaR FOR {len(df_covar)} BANKS")
    print("="*80)
    print(f"Data saved to: {output_file}")
    
    print("\n🚨 Top 5 Most Systemically Dangerous Banks by ΔCoVaR (More negative = Worse):")
    print(df_covar.head(5).to_string(index=False))
else:
    print("❌ No ΔCoVaR metrics calculated.")