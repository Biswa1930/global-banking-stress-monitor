import pandas as pd
import numpy as np
import torch
from pathlib import Path

def calculate_ewma_covariance(returns_df, lambda_decay=0.94):
    """
    Computes time-varying covariance matrices using the RiskMetrics EWMA approach.
    Acts as a robust DCC-GARCH proxy for large multi-node networks.
    
    Parameters:
    - returns_df: DataFrame of daily log returns (T x N)
    - lambda_decay: Decay factor (0.94 is the RiskMetrics standard for daily data)
    
    Returns:
    - covariance_tensor: numpy array of shape (T, N, N)
    """
    # Extract raw numpy arrays
    dates = returns_df.index
    data = returns_df.values
    T, N = data.shape
    
    # Initialize the tensor to hold T covariance matrices of size N x N
    cov_tensor = np.zeros((T, N, N))
    
    # Initialize day 0 with the unconditional sample covariance
    cov_tensor[0] = np.cov(data, rowvar=False)
    
    # Recursively calculate the EWMA covariance for each day
    for t in range(1, T):
        # Current return vector (column format)
        r_t = data[t].reshape(-1, 1)
        
        # Update rule: Cov_t = lambda * Cov_{t-1} + (1 - lambda) * (r_t * r_t^T)
        cov_tensor[t] = lambda_decay * cov_tensor[t-1] + (1 - lambda_decay) * (r_t @ r_t.T)
        
    return cov_tensor, dates

def covariance_to_correlation(cov_tensor):
    """
    Converts a tensor of covariance matrices into correlation matrices.
    These correlation values act as the dynamic edge weights for the PIDL model.
    """
    T, N, _ = cov_tensor.shape
    corr_tensor = np.zeros_like(cov_tensor)
    
    for t in range(T):
        cov_matrix = cov_tensor[t]
        
        # Extract variances (diagonal elements) and COPY to make it mutable
        variances = np.diag(cov_matrix).copy()
        
        # Handle zeros to prevent division by zero
        variances[variances == 0] = 1e-8
        
        # Calculate standard deviations
        std_devs = np.sqrt(variances)
        
        # Calculate correlation matrix: Corr_{i,j} = Cov_{i,j} / (Std_i * Std_j)
        std_matrix = np.outer(std_devs, std_devs)
        corr_tensor[t] = cov_matrix / std_matrix
        
        # Ensure numerical stability bounded strictly between -1 and 1
        corr_tensor[t] = np.clip(corr_tensor[t], -1.0, 1.0)
        
    return corr_tensor
if __name__ == "__main__":
    print("🕸️ Initializing Phase 3B: Dynamic Conditional Correlation (EWMA) Network...")
    
    # Dynamically resolve absolute paths based on the script's location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "processed" / "daily_returns.csv"
    
    if not data_path.exists():
        print(f"Error: Could not find {data_path}. Run market_data_pipeline.py first.")
    else:
        df_returns = pd.read_csv(data_path)
        
        # Set Date as index and drop the System column (we only want bank-to-bank edges)
        df_returns['Date'] = pd.to_datetime(df_returns['Date'])
        df_returns.set_index('Date', inplace=True)
        
        if '^GSPC' in df_returns.columns:
            df_banks = df_returns.drop(columns=['^GSPC'])
        else:
            df_banks = df_returns
            
        print(f"Processing dynamically varying edges for {df_banks.shape[1]} banks over {df_banks.shape[0]} trading days...")
        
        # Compute the dynamic covariance and correlation tensors
        cov_tensor, dates = calculate_ewma_covariance(df_banks, lambda_decay=0.94)
        corr_tensor = covariance_to_correlation(cov_tensor)
        
        # Convert to PyTorch Tensor for Phase 3 ingestion
        torch_corr_tensor = torch.tensor(corr_tensor, dtype=torch.float32)
        
        # Save the tensor dynamically
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / "dynamic_edges_tensor.pt"
        torch.save(torch_corr_tensor, tensor_path)
        
        print("-" * 60)
        print("✅ DYNAMIC NETWORK GENERATED SUCCESSFULLY")
        print(f"Tensor Shape: {torch_corr_tensor.shape} (Days x Banks x Banks)")
        print(f"File saved to: {tensor_path.resolve()}")
        print("Ready for walk-forward training loop integration.")
        print("-" * 60)