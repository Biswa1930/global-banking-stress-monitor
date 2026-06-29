import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

print("Initializing Physics-Informed Deep Learning (PIDL) Engine via PyTorch...\n")

try:
    srisk_df = pd.read_csv("data/srisk_capital_shortfalls.csv")

    # Normalize our core engineering metrics as inputs
    scaler = MinMaxScaler()
    X_raw = np.column_stack([
        srisk_df['GARCH_Beta'].values,
        srisk_df['LRMES_Pct'].values,
        srisk_df['Capital_Shortfall_SRISK_Billions'].values
    ])
    
    X_train_np = scaler.fit_transform(X_raw).astype(np.float32)
    # Target proxy: Empirical Systemic Stress Level
    y_train_np = (0.4 * X_train_np[:, 0] + 0.6 * X_train_np[:, 2]).reshape(-1, 1)

    # Convert to PyTorch tensors. We require gradients on inputs to enforce physical bounds.
    X_train = torch.tensor(X_train_np, requires_grad=True)
    y_train = torch.tensor(y_train_np)

except FileNotFoundError:
    print("❌ Critical Error: data/srisk_capital_shortfalls.csv not found.")
    exit()

# 1. Define the Physics-Informed Deep Learning Network Architecture
class PIDLStressNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = PIDLStressNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Beginning Gradient Descent via Custom Physics Constraints...")

epochs = 200
lambda_physics = 1.5

# 2. Neural Network Custom Training Loop
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # Forward pass
    predictions = model(X_train)

    # Standard Data-Driven Loss (MSE)
    mse_loss = torch.mean((predictions - y_train) ** 2)

    # Physics Penalty: Enforcing No-Arbitrage and Monotonicity Constraints
    # Compute the Jacobian (gradients of output stress w.r.t input features)
    gradients = torch.autograd.grad(
        outputs=predictions,
        inputs=X_train,
        grad_outputs=torch.ones_like(predictions),
        create_graph=True
    )[0]

    # Financial Law constraint: Systemic stress must strictly increase as individual risk metrics increase. 
    # torch.relu(-gradients) isolates negative gradients and squares them to penalize the network.
    physics_penalty = torch.mean(torch.relu(-gradients) ** 2)

    # Total Composite Loss
    total_loss = mse_loss + (lambda_physics * physics_penalty)

    # Backward pass & optimize
    total_loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.6f} | MSE: {mse_loss.item():.6f} | Physics Penalty: {physics_penalty.item():.6f}")

# 3. Synthesize and Export Final Unified Systemic Stress Index
with torch.no_grad():
    final_stress_scores = model(X_train).numpy().flatten()

srisk_df['Unified_Systemic_Stress_Index'] = np.round(final_stress_scores * 100, 2)
output_df = srisk_df.sort_values(by='Unified_Systemic_Stress_Index', ascending=False).reset_index(drop=True)

output_df.to_csv("data/systemic_risk_rankings.csv", index=False)

print("\n==================================================")
print("   UNIFIED SYSTEMIC RISK RANKINGS (PIDL SYNTHESIS)  ")
print("==================================================")
print(output_df[['Bank', 'Capital_Shortfall_SRISK_Billions', 'Unified_Systemic_Stress_Index']].to_string(index=True))
print("==================================================")
print("\n✅ PIDL synthesis complete. Final rankings saved to: data/systemic_risk_rankings.csv")