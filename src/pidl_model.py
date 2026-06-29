import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class PIDLStressEngine(nn.Module):
    """
    Physics-Informed Deep Learning (PIDL) Engine for synthesizing 
    macro-prudential capital shortfalls and network contagion infectivity.
    """
    def __init__(self, input_dim=2):
        super(PIDLStressEngine, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train_pidl_model(features, epochs=200, lr=0.01):
    """
    Trains the PIDL network using a dual optimization objective:
    min (MSE Loss + Physics Bound Penalties)
    """
    X_train = torch.tensor(features, dtype=torch.float32)
    
    model = PIDLStressEngine(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    
    # Target proxy for optimization demonstration
    y_pseudo = torch.tensor(features[:, 0] * 0.6 + features[:, 1] * 0.4, dtype=torch.float32).unsqueeze(1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train)
        mse_loss = mse_criterion(predictions, y_pseudo)
        
        # Macro-prudential Boundary Constraint (Physics Penalty)
        # Assures that index outputs never fall below zero under positive stress features
        physics_penalty = torch.mean(torch.relu(-predictions))
        
        total_loss = mse_loss + 1.5 * physics_penalty
        
        total_loss.backward()
        optimizer.step()
        
    return model