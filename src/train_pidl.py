import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from pathlib import Path
import warnings

# Import the architecture and curvature math
from pidl_model import PIDLModel, compute_ruppeiner_curvature

warnings.filterwarnings('ignore')

def safe_fokker_planck_loss(recon_x, x, mu, log_var, lambda_fp=0.1):
    """
    An audit-grade, numerically stable Fokker-Planck Divergence loss.
    Prevents float32 exponential overflow during early training epochs.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    log_var_clamped = torch.clamp(log_var, min=-20, max=20)
    fp_divergence = -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp())
    fp_divergence = fp_divergence / x.size(0)
    
    total_loss = recon_loss + (lambda_fp * fp_divergence)
    return total_loss, recon_loss, fp_divergence

def load_and_prepare_tensors(project_root):
    """
    Loads data, standardizes it, and ruthlessly scrubs NaN/Inf values.
    """
    returns_path = project_root / "data" / "processed" / "daily_returns.csv"
    df_returns = pd.read_csv(returns_path)
    df_returns['Date'] = pd.to_datetime(df_returns['Date'])
    df_returns.set_index('Date', inplace=True)
    
    if '^GSPC' in df_returns.columns:
        df_banks = df_returns.drop(columns=['^GSPC'])
    else:
        df_banks = df_returns
        
    df_banks = df_banks.ffill().fillna(0)
        
    edges_path = project_root / "data" / "processed" / "dynamic_edges_tensor.pt"
    corr_tensor = torch.load(edges_path)
    
    returns_tensor = torch.tensor(df_banks.values, dtype=torch.float32)
    node_degrees = torch.sum(torch.abs(corr_tensor), dim=2) 
    raw_input_tensor = torch.cat([returns_tensor, node_degrees], dim=1)
    
    mean = raw_input_tensor.mean(dim=0, keepdim=True)
    std = raw_input_tensor.std(dim=0, keepdim=True)
    input_tensor = (raw_input_tensor - mean) / (std + 1e-8)
    
    input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
    
    return input_tensor, df_banks.index, df_banks.columns

if __name__ == "__main__":
    print("🚀 Initializing Phase 3C: Walk-Forward PIDL Training Loop (Mini-Batch)...")
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    input_tensor, dates, banks = load_and_prepare_tensors(project_root)
    T, input_dim = input_tensor.shape
    
    print(f"Loaded T={T} trading days. Feature dimensionality: {input_dim}")
    
    train_end = int(T * 0.6)
    val_end = int(T * 0.8)
    
    train_data = input_tensor[:train_end]
    val_data = input_tensor[train_end:val_end]
    test_data = input_tensor[val_end:]
    
    batch_size = 32
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Split: Train ({len(train_data)}), Validation ({len(val_data)}), Test ({len(test_data)})")
    
    # ---------------------------------------------------------
    # FINE TUNING: Latent Dim = 4 | Lambda FP = 0.005
    # ---------------------------------------------------------
    model = PIDLModel(input_dim=input_dim, hidden_dim=64, latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    epochs = 150
    lambda_fp = 0.005 
    
    print("\nStarting Training Loop...")
    print("-" * 65)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            batch_x = batch[0]
            optimizer.zero_grad()
            
            recon_x, mu, log_var, latent_z = model(batch_x)
            loss, mse, fp_div = safe_fokker_planck_loss(recon_x, batch_x, mu, log_var, lambda_fp)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_recon, v_mu, v_log_var, v_z = model(val_data)
                v_loss, v_mse, v_fp = safe_fokker_planck_loss(val_recon, val_data, v_mu, v_log_var, lambda_fp)
                R = compute_ruppeiner_curvature(v_z)
                print(f"Epoch {epoch:<3} | Train Loss: {avg_train_loss:<8.4f} | Val Loss: {v_loss.item():<8.4f} | R: {R:<8.4f}")

    model_path = project_root / "data" / "processed" / "pidl_trained_model.pt"
    torch.save(model.state_dict(), model_path)
    
    print("-" * 65)
    print("✅ WALK-FORWARD TRAINING COMPLETE")