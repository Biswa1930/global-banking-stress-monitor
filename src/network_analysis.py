import pandas as pd
import networkx as nx
import os

print("🌐 Initializing Phase 2A: Network Centrality Engine...")

# 1. Define Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
edges_file = os.path.join(processed_dir, 'network_edges.csv')
output_file = os.path.join(processed_dir, 'network_centrality.csv')

# 2. Load the Edge List
if not os.path.exists(edges_file):
    raise FileNotFoundError(f"❌ Missing {edges_file}. Please run the Phase 1D builder first.")

df_edges = pd.read_csv(edges_file)

# Filter out zero exposures to keep the graph sparse and accurate
df_edges = df_edges[df_edges['Exposure_Billion_USD'] > 0]

print(f"📊 Loaded {len(df_edges)} active interbank lending edges.")

# 3. Build the Directed Graph
# DiGraph because lending is directional (Lender -> Borrower)
G = nx.from_pandas_edgelist(
    df_edges, 
    source='Lender_Ticker', 
    target='Borrower_Ticker', 
    edge_attr='Exposure_Billion_USD', 
    create_using=nx.DiGraph()
)

print("🧮 Calculating systemic topology metrics (NetworkX)...")

# 4. Calculate Centrality Measures
# We use weight='Exposure_Billion_USD' where applicable so a $400B edge matters more than a $1B edge.

# A. Degree Centrality (Connectivity)
in_degree = dict(G.in_degree(weight='Exposure_Billion_USD'))
out_degree = dict(G.out_degree(weight='Exposure_Billion_USD'))

# B. PageRank (Systemic Importance - Who is connected to other highly connected banks?)
pagerank = nx.pagerank(G, weight='Exposure_Billion_USD')

# C. Betweenness Centrality (Bottleneck Risk - Who sits on the shortest paths?)
# Note: For a star graph, SYSTEM will have 1.0, banks will have 0.0. 
# This will dynamically change when bilateral data is added.
betweenness = nx.betweenness_centrality(G, weight='Exposure_Billion_USD')

# 5. Compile the Results
results = []
for node in G.nodes():
    if node == 'SYSTEM': # We only care about the banks, not the placeholder node
        continue
        
    results.append({
        'Bank_Ticker': node,
        'Out_Exposure_Billion': round(out_degree.get(node, 0), 2),
        'In_Exposure_Billion': round(in_degree.get(node, 0), 2),
        'PageRank_Score': round(pagerank.get(node, 0), 4),
        'Betweenness_Centrality': round(betweenness.get(node, 0), 4)
    })

df_centrality = pd.DataFrame(results)

# Sort by systemic footprint (Out_Exposure)
if not df_centrality.empty:
    df_centrality = df_centrality.sort_values(by='Out_Exposure_Billion', ascending=False)
    df_centrality.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print(f"✅ SUCCESSFULLY CALCULATED NETWORK CENTRALITY ({len(df_centrality)} Active Banks)")
    print("="*80)
    print(f"Data saved to: {output_file}")
    
    print("\n🔍 Top 5 Most Systemically Connected Banks (Current Topology):")
    print(df_centrality.head(5).to_string(index=False))
else:
    print("❌ No valid bank nodes found in the network.")