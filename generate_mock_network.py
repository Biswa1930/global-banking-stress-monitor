import pandas as pd
import numpy as np
import os

# The 8 active US G-SIBs
banks = ["JPM", "BAC", "C", "WFC", "GS", "MS", "BK", "STT"]
edges = []

np.random.seed(42) # For reproducibility

print("Generating proxy interbank exposure network...")

for lender in banks:
    for borrower in banks:
        if lender != borrower:
            # Simulate exposure in billions USD. 
            # Probability of a link is 70% to simulate a dense G-SIB network.
            if np.random.rand() > 0.3:
                # Base exposure between $1B and $15B
                exposure = np.random.uniform(1.0, 15.0)
                
                # JPM and BAC act as massive liquidity hubs, so we boost their exposures
                if lender in ["JPM", "BAC"] or borrower in ["JPM", "BAC"]:
                    exposure *= 2.5 
                    
                edges.append({
                    "lender": lender,
                    "borrower": borrower,
                    "exposure": round(exposure, 2)
                })

df_edges = pd.DataFrame(edges)

# Save to the processed data folder
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/network_edges.csv"
df_edges.to_csv(output_path, index=False)

print(f"✅ Generated {len(df_edges)} directed interbank links.")
print(f"Saved to: {output_path}")
print("Ready for Phase 2A: Network Topology Analysis.")