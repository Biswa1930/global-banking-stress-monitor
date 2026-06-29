"""
Market Microstructure Risk Metrics - Global Banking Stress Monitor
------------------------------------------------------------------
Computes marginal systemic risk contributions (ΔCoVaR) and 
structural capital shortfalls (SRISK) during severe market shocks.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def compute_srisk(equity_val: float, debt_val: float, lrmes: float, k: float = 0.08) -> float:
    """
    Computes the SRISK (Capital Shortfall) of a bank in billions.
    SRISK = E[Capital Shortfall | Systemic Crisis]
    
    Args:
        equity_val (float): Current market capitalization.
        debt_val (float): Book value of total liabilities.
        lrmes (float): Long-Run Marginal Expected Shortfall (expected equity drop).
        k (float): Prudential capital ratio requirement (default 8%).
        
    Returns:
        float: Expected capital shortfall in currency units. Returns 0 if well-capitalized.
    """
    # Formula: SRISK = k * Debt - (1 - k) * Equity * (1 - LRMES)
    # Note: LRMES is input as a positive decimal (e.g., 0.40 for a 40% drop)
    
    expected_equity_post_shock = equity_val * (1 - lrmes)
    required_capital = k * (debt_val + expected_equity_post_shock)
    
    srisk = required_capital - expected_equity_post_shock
    
    # If SRISK is negative, the bank has a capital surplus, so shortfall is 0
    return max(srisk, 0.0)

def compute_delta_covar(df_returns: pd.DataFrame, target_bank: str, system_index: str = '^GSPC', q: float = 0.05) -> float:
    """
    Calculates ΔCoVaR via Quantile Regression (Adrian & Brunnermeier).
    ΔCoVaR measures the change in system-wide VaR when the target bank shifts
    from its median state to its distressed state (e.g., 5th percentile).
    
    Args:
        df_returns (pd.DataFrame): Matrix of log returns for banks and the market.
        target_bank (str): Column name of the bank being stressed.
        system_index (str): Column name representing the market/system.
        q (float): The distress quantile (default 5% / 0.05).
        
    Returns:
        float: The ΔCoVaR value.
    """
    temp_df = df_returns[[system_index, target_bank]].dropna()
    temp_df.columns = ['system', 'bank']
    
    try:
        # Fit 5th percentile regression (Distressed State)
        mod_distress = smf.quantreg('system ~ bank', temp_df)
        res_distress = mod_distress.fit(q=q)
        
        # Fit 50th percentile regression (Median/Normal State)
        mod_median = smf.quantreg('system ~ bank', temp_df)
        res_median = mod_median.fit(q=0.50)
        
        # Calculate bank's specific VaR limits
        var_distress = temp_df['bank'].quantile(q)
        var_median = temp_df['bank'].median()
        
        # CoVaR = Alpha + Beta * Bank_VaR
        covar_distress = res_distress.params['Intercept'] + res_distress.params['bank'] * var_distress
        covar_median = res_median.params['Intercept'] + res_median.params['bank'] * var_median
        
        # ΔCoVaR is the difference between the system's risk when bank is distressed vs normal
        delta_covar = covar_distress - covar_median
        
        return delta_covar
        
    except Exception as e:
        print(f"⚠️ Quantile regression failed for {target_bank}: {e}")
        return np.nan