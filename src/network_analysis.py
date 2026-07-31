import pandas as pd
import numpy as np
import networkx as nx
import os

print("🕸️ Initializing Phase 2A: Systemic Network Topology (Gravity Model)...")

# ==========================================
# 1. PATHS & PIPELINE HANDOFF
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')

# Enforce Single Source of Truth: Load the output from data_ingestion.py
clean_data_path = os.path.join(processed_dir, 'cleaned_bank_nodes.csv')
if not os.path.exists(clean_data_path):
    raise FileNotFoundError(f"Pipeline Error: Run data_ingestion.py first. Missing: {clean_data_path}")

df_nodes = pd.read_csv(clean_data_path)
df_nodes.set_index('Bank_Ticker', inplace=True)
tickers = df_nodes.index.tolist()

# ==========================================
# 2. THE GRAVITY MODEL (EDGE CONSTRUCTION)
# ==========================================
print(f"🪐 Distributing Interbank Exposure across {len(tickers)} Global Hubs...")

edges = []
total_system_assets = df_nodes['Total_Assets_Billion'].sum()

for lender in tickers:
    # Use .get() to handle potential missing columns gracefully, defaulting to 0 if NaN
    total_exposure = df_nodes.loc[lender, 'Interbank_Exposure_Billion']
    if pd.isna(total_exposure) or total_exposure == 0:
        continue
    
    for borrower in tickers:
        if lender == borrower:
            continue
            
        # Borrower's gravitational pull = (Borrower Assets / All Other Assets)
        borrower_assets = df_nodes.loc[borrower, 'Total_Assets_Billion']
        assets_excluding_lender = total_system_assets - df_nodes.loc[lender, 'Total_Assets_Billion']
        
        gravity_weight = borrower_assets / assets_excluding_lender
        bilateral_exposure = total_exposure * gravity_weight
        
        edges.append({
            'Lender_Ticker': lender,
            'Borrower_Ticker': borrower,
            'Exposure_Billion_USD': round(bilateral_exposure, 3)
        })

df_edges = pd.DataFrame(edges)

# ==========================================
# 3. NETWORK MATH & CENTRALITY
# ==========================================
print("🧮 Calculating PageRank & Eigenvector Centralities...")

# Build a directed weighted Graph
G = nx.DiGraph()

for _, row in df_edges.iterrows():
    if row['Exposure_Billion_USD'] > 0:
        G.add_edge(row['Lender_Ticker'], row['Borrower_Ticker'], weight=row['Exposure_Billion_USD'])

# Calculate Network Centrality Metrics
try:
    pagerank = nx.pagerank(G, weight='weight')
    
    # Ensure convergence on dense matrices
    eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
    betweenness = nx.betweenness_centrality(G, weight='weight')
except Exception as e:
    print(f"⚠️ Network math warning: {e}")
    pagerank, eigenvector, betweenness = {}, {}, {}

# Compile Node Centrality DataFrame
centrality_data = []
for bank in tickers:
    out_exposure = df_edges[df_edges['Lender_Ticker'] == bank]['Exposure_Billion_USD'].sum() if not df_edges.empty else 0
    in_exposure = df_edges[df_edges['Borrower_Ticker'] == bank]['Exposure_Billion_USD'].sum() if not df_edges.empty else 0
    
    centrality_data.append({
        'Bank_Ticker': bank,
        'Out_Exposure_Billion': round(out_exposure, 2),
        'In_Exposure_Billion': round(in_exposure, 2),
        'PageRank_Score': round(pagerank.get(bank, 0.0), 4),
        'Eigenvector_Score': round(eigenvector.get(bank, 0.0), 4),
        'Betweenness_Score': round(betweenness.get(bank, 0.0), 4)
    })

df_centrality = pd.DataFrame(centrality_data).sort_values(by='PageRank_Score', ascending=False)

# ==========================================
# 4. EXPORT TO PROCESSED DIRECTORY
# ==========================================
output_edges = os.path.join(processed_dir, 'network_edges.csv')
output_nodes = os.path.join(processed_dir, 'network_centrality.csv')

df_edges.to_csv(output_edges, index=False)
df_centrality.to_csv(output_nodes, index=False)

print("\n" + "="*80)
print(f"✅ SUCCESSFULLY MAPPED SYSTEMIC NETWORK")
print("="*80)
print(f"Edges Generated: {len(df_edges)}")
print("\nTop 5 Global Hubs (by PageRank):")
print(df_centrality.head(5)[['Bank_Ticker', 'Out_Exposure_Billion', 'PageRank_Score']].to_string(index=False))