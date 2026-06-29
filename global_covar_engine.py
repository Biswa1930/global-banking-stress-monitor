import pandas as pd
import statsmodels.formula.api as smf
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/aligned_market_data.csv"

print("Igniting Global Delta CoVaR Engine...\n")

try:
    df = pd.read_csv(DATA_PATH)
    
    # 1. Define our target banks (excluding our macro controls)
    # This automatically finds all the bank columns we merged earlier
    macro_cols = ['date', 'Market_Return_Proxy', 'VIX_Volatility', 'Treasury_10Y_Yield', 'Treasury_3M_Yield', 'Term_Spread', 'Market_Return']
    banks = [col for col in df.columns if col not in macro_cols and not col.endswith('_Return')]
    
    results = []
    
    print(f"Executing quantile regressions for {len(banks)} Global Systemically Important Banks...")
    
    # 2. Loop through every single bank
    for bank in banks:
        # Create a temporary return column for the current bank in the loop
        df['Bank_Return'] = df[bank].pct_change()
        
        # We must drop NaNs dynamically for each loop to avoid regression crashes
        temp_df = df.dropna(subset=['Bank_Return', 'Market_Return', 'VIX_Volatility', 'Term_Spread', 'Treasury_3M_Yield']).copy()
        
        # Model 1: Bank's own risk
        bank_formula = "Bank_Return ~ VIX_Volatility + Term_Spread + Treasury_3M_Yield"
        res_bank_5 = smf.quantreg(bank_formula, temp_df).fit(q=0.05)
        res_bank_50 = smf.quantreg(bank_formula, temp_df).fit(q=0.50)
        
        temp_df['VaR'] = res_bank_5.predict(temp_df)
        temp_df['Median'] = res_bank_50.predict(temp_df)
        
        # Model 2: Market's risk (CoVaR)
        market_formula = "Market_Return ~ Bank_Return + VIX_Volatility + Term_Spread + Treasury_3M_Yield"
        res_market_5 = smf.quantreg(market_formula, temp_df).fit(q=0.05)
        
        market_sensitivity = res_market_5.params['Bank_Return']
        
        # Model 3: Calculate Delta CoVaR
        temp_df['Delta_CoVaR'] = market_sensitivity * (temp_df['VaR'] - temp_df['Median'])
        
        avg_delta_covar_pct = temp_df['Delta_CoVaR'].mean() * 100
        
        results.append({
            "Bank": bank,
            "Market_Sensitivity_Gamma": round(market_sensitivity, 4),
            "Delta_CoVaR_Pct": round(avg_delta_covar_pct, 3)
        })

    # 3. Rank them from most dangerous (lowest negative number) to least dangerous
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Delta_CoVaR_Pct", ascending=True).reset_index(drop=True)
    
    # Save the final results to your data folder
    results_df.to_csv("data/systemic_risk_rankings.csv", index=False)
    
    print("\n==================================================")
    print(" GLOBAL SYSTEMIC RISK RANKINGS (\u0394CoVaR) ")
    print("==================================================")
    print(results_df.to_string(index=True))
    print("==================================================")
    print("\n✅ Master Rankings saved to: data/systemic_risk_rankings.csv")

except Exception as e:
    print(f"❌ Error in master engine: {e}")