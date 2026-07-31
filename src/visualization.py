import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pathlib import Path
import warnings

from pidl_model import PIDLModel, compute_ruppeiner_curvature
from train_pidl import load_and_prepare_tensors 

warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid")

def generate_rolling_curvature(latent_states, window=21):
    T = latent_states.shape[0]
    rolling_R = np.full(T, np.nan)
    
    for i in range(window, T):
        window_z = latent_states[i-window:i]
        R = compute_ruppeiner_curvature(window_z)
        rolling_R[i] = R
        
    return rolling_R

if __name__ == "__main__":
    print("📊 Initializing Phase 4: Systemic Risk Inference & Visualization...")
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    print("Loading banking data and standardizing inputs...")
    input_tensor, dates, banks = load_and_prepare_tensors(project_root)
    
    print("Fetching S&P 500 benchmark data...")
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = (dates.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    sp500_ticker = yf.Ticker('^GSPC')
    sp500_df = sp500_ticker.history(start=start_date, end=end_date)
    
    sp500_close = sp500_df['Close']
    sp500_close.index = sp500_close.index.tz_localize(None)
        
    sp500_returns = sp500_close.pct_change().reindex(dates).fillna(0)
    sp500_cum_returns = (1 + sp500_returns).cumprod()
    
    print("Loading trained PIDL network...")
    input_dim = input_tensor.shape[1]
    
    # Matches the updated 4 dimensions from the training script
    model = PIDLModel(input_dim=input_dim, hidden_dim=64, latent_dim=4) 
    
    model_path = project_root / "data" / "processed" / "pidl_trained_model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    
    print("Extracting latent thermodynamic states...")
    with torch.no_grad():
        _, _, _, latent_z = model(input_tensor)
        
    print("Calculating rolling Ruppeiner Curvature (R)...")
    rolling_R = generate_rolling_curvature(latent_z, window=21)
    
    print("Generating visual dashboard...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(dates, sp500_cum_returns, color='blue', linewidth=1.5, label='S&P 500 Cumulative Return')
    ax1.set_title("Global Banking Stress Monitor: PIDL Early Warning System", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Market Benchmark", fontsize=12)
    ax1.legend(loc="upper left")
    
    clipped_R = np.clip(rolling_R, -5, 5) 
    
    ax2.plot(dates, clipped_R, color='red', linewidth=1.2, label='Ruppeiner Curvature (R)')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(dates, 0, clipped_R, where=(clipped_R > 0), color='red', alpha=0.3, label='Critical Phase (Fragility)')
    ax2.fill_between(dates, clipped_R, 0, where=(clipped_R <= 0), color='green', alpha=0.3, label='Stable Phase')
    
    ax2.set_ylabel("Thermodynamic Curvature", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    
    output_dir = project_root / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "systemic_risk_dashboard.png"
    plt.savefig(out_file, dpi=300)
    
    print("-" * 65)
    print("✅ PIPELINE COMPLETE")
    print(f"Dashboard saved to: {out_file.resolve()}")