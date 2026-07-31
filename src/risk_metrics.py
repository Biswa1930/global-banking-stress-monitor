import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import os

def calculate_delta_covar(df_returns, bank_col, system_col):
    """
    Implements ΔCoVaR from scratch using quantile regression (Adrian & Brunnermeier 2016).
    Formula: ΔCoVaR_i = CoVaR(system | bank_i in distress) - CoVaR(system | bank_i at median)
    """
    # Create a temporary dataframe for the regression
    data = pd.DataFrame({
        'system': df_returns[system_col],
        'bank': df_returns[bank_col]
    }).dropna()
    
    if len(data) < 250: # Ensure enough trading days for stable quantiles
        return np.nan
        
    # 1.Regress system returns on bank returns at the 5th percentile (Distress)
    mod_distress = smf.quantreg('system ~ bank', data)
    res_distress = mod_distress.fit(q=0.05, max_iter=5000, p_tol=1e-5)
    
    # 2. Regress system returns on bank returns at the 50th percentile (Median state)
    mod_median = smf.quantreg('system ~ bank', data)
    res_median = mod_median.fit(q=0.50, max_iter=5000, p_tol=1e-5)
    
    # 3. Calculate the Bank's Value at Risk (VaR) at 5% and 50%
    var_5 = np.percentile(data['bank'], 5)
    var_50 = np.percentile(data['bank'], 50)
    
    # 4. Compute CoVaR (System risk conditional on bank state)
    covar_distress = res_distress.params['Intercept'] + res_distress.params['bank'] * var_5
    covar_median = res_median.params['Intercept'] + res_median.params['bank'] * var_50
    
    # 5. ΔCoVaR is the difference
    delta_covar = covar_distress - covar_median
    return delta_covar

def calculate_srisk(equity_value, debt_value, lrmes, prudential_ratio=0.08):
    """
    Implements SRISK (Brownlees & Engle 2017).
    Formula: SRISK_i = E_i * (k - (1-k) * LRMES_i)
    """
    k = prudential_ratio
    # LRMES represents the Long-Run Marginal Expected Shortfall (expected drop in equity during a 40% market crash)
    capital_shortfall = (debt_value * k) - ((1 - k) * equity_value * (1 - lrmes))
    
    # SRISK is only recognized if the capital shortfall is positive (bank needs capital)
    return max(0, capital_shortfall)

def run_risk_metrics_pipeline(returns_file, balance_sheet_file, output_dir):
    print("📈 Initializing Phase 2B: Market Microstructure Risk Metrics...")
    
    returns_path = Path(returns_file)
    bs_path = Path(balance_sheet_file)
    
    if not returns_path.exists() or not bs_path.exists():
        print("⚠️ Waiting on Market Data (Phase 1B). Feed daily returns to compute CoVaR/SRISK.")
        return
        
    df_returns = pd.read_csv(returns_path, parse_dates=['Date']).set_index('Date')
    df_bs = pd.read_csv(bs_path)
    
    results = []
    
    print("Executing Quantile Regressions for ΔCoVaR...")
    # Assuming 'SPX' or 'MSCI_World' is your system return column
    system_ticker = 'SYSTEM_RETURN' 
    
    for _, row in df_bs.iterrows():
        ticker = row['Bank_Ticker']
        if ticker in df_returns.columns:
            # 1. ΔCoVaR
            d_covar = calculate_delta_covar(df_returns, ticker, system_ticker)
            
            # 2. SRISK (Mocking LRMES here; DCC-GARCH will provide exact LRMES next)
            # Proxying LRMES as a stressed historical beta for the scaffolding
            historical_beta = df_returns[ticker].cov(df_returns[system_ticker]) / df_returns[system_ticker].var()
            proxied_lrmes = min(0.99, historical_beta * 0.40) # 40% market decline shock
            
            # Extract balance sheet constraints (Assuming these were pulled in Phase 1)
            equity = row.get('Market_Cap_Billion', row['Total_Assets_Billion'] * 0.10) 
            debt = row['Total_Assets_Billion'] - equity
            
            srisk_val = calculate_srisk(equity, debt, proxied_lrmes)
            
            results.append({
                'Bank_Ticker': ticker,
                'Delta_CoVaR': round(d_covar, 4) if pd.notna(d_covar) else np.nan,
                'SRISK_Billion': round(srisk_val, 2)
            })
            
    df_results = pd.DataFrame(results)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(output_dir) / "market_risk_metrics.csv"
    df_results.to_csv(out_file, index=False)
    
    print("-" * 50)
    print("TOP 3 MOST VULNERABLE BANKS (by ΔCoVaR):")
    print(df_results.nsmallest(3, 'Delta_CoVaR')[['Bank_Ticker', 'Delta_CoVaR']].to_string(index=False))
    print("-" * 50)
    print(f"Phase 2B Output saved to: {out_file.resolve()}\n")

# --- Execution ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # These will be the inputs from your yfinance pipeline (Phase 1B)
    RETURNS_FILE = os.path.join(script_dir, '..', 'data', 'processed', 'daily_returns.csv')
    BS_FILE = os.path.join(script_dir, '..', 'data', 'processed', 'cleaned_bank_nodes.csv')
    OUTPUT_DIR = os.path.join(script_dir, '..', 'data', 'processed')
    
    run_risk_metrics_pipeline(RETURNS_FILE, BS_FILE, OUTPUT_DIR)