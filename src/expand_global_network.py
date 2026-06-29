import pandas as pd
import os

print("🌍 Initializing Phase 1D: Global Network Expansion (European Bloc)...")

# 1. Locate the dynamic data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

nodes_path = os.path.join(data_dir, 'network_nodes.csv')
edges_path = os.path.join(data_dir, 'network_edges.csv')

# Load existing US Data
df_nodes = pd.read_csv(nodes_path)
df_edges = pd.read_csv(edges_path)

# 2. The European G-SIB Nodes (2023 End-of-Year 10-K / Pillar 3 Data in $ Billions)
euro_nodes = pd.DataFrame({
    'bank_id': ['HSBC', 'BCS', 'DB', 'BNPQY', 'SCGLY', 'UBS', 'ING', 'SAN', 'UNCFF', 'NRDEF'],
    'total_assets_bln': [3038.0, 1894.0, 1422.0, 2855.0, 1690.0, 1717.0, 1070.0, 1950.0, 930.0, 635.0],
    'total_equity_bln': [198.0, 89.0, 78.0, 131.0, 71.0, 85.0, 62.0, 105.0, 68.0, 32.0],
    'tier_1_capital_ratio': [14.8, 13.5, 13.7, 13.2, 13.1, 14.4, 15.3, 12.3, 15.8, 17.0]
})

# 3. The European Edges (Estimated Cross-Border & Domestic Financial Exposure)
# Note: European banks disclose differently than US banks; these represent Wholesale Financial exposures
euro_edges = pd.DataFrame({
    'lender': ['HSBC', 'BCS', 'DB', 'BNPQY', 'SCGLY', 'UBS', 'ING', 'SAN', 'UNCFF', 'NRDEF'],
    'borrower': ['TOTAL_FINANCIALS'] * 10,
    'exposure_bln': [142.5, 95.2, 110.4, 135.0, 88.6, 120.1, 45.3, 65.8, 41.2, 28.5]
})

# 4. Merge and Save
df_nodes_global = pd.concat([df_nodes, euro_nodes], ignore_index=True)
df_edges_global = pd.concat([df_edges, euro_edges], ignore_index=True)

df_nodes_global.to_csv(nodes_path, index=False)
df_edges_global.to_csv(edges_path, index=False)

print(f"✅ Successfully added 10 European G-SIBs to the network.")
print(f"📊 Global Network now contains {len(df_nodes_global)} mega-banks.")
print("Run `visualize_network.py` to see the new global topology!")