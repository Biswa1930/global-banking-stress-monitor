import pandas as pd
import os

# 1. Locate the data folder dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

print("🏗️ Building the missing Edges Database...")

# 2. The exposure data we extracted manually from the SEC 10-K filings
edges_data = {
    'lender': ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'STT', 'BK'],
    'borrower': ['TOTAL_FINANCIALS'] * 8,
    'exposure_bln': [57.17, 85.66, 83.51, 158.45, 34.28, 73.45, 33.56, 39.70]
}

# 3. Save it directly to the data/ folder
df_edges = pd.DataFrame(edges_data)
edges_path = os.path.join(data_dir, 'network_edges.csv')
df_edges.to_csv(edges_path, index=False)

print(f"✅ Saved network_edges.csv successfully to:\n{edges_path}")