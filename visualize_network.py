import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

print("🕸️ Initializing Systemic Network Topology...")

# --- BULLETPROOF PATH HANDLING ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check if we are in the root folder or the src folder
if os.path.exists(os.path.join(script_dir, 'data')):
    data_dir = os.path.join(script_dir, 'data') # Script is in root
else:
    data_dir = os.path.join(script_dir, '..', 'data') # Script is in src/

edges_path = os.path.join(data_dir, 'network_edges.csv')
nodes_path = os.path.join(data_dir, 'network_nodes.csv')

# 1. Load your proprietary datasets from the data/ folder
try:
    df_edges = pd.read_csv(edges_path)
    df_nodes = pd.read_csv(nodes_path)
except FileNotFoundError:
    print(f"❌ Error: Could not find CSV files in {data_dir}.")
    exit()

# 2. Initialize a Directed Graph
G = nx.DiGraph()

# 3. Add the central aggregate node
G.add_node("TOTAL_FINANCIALS", size=5000, label="Financial\nSystem")

# 4. Add the Bank nodes and scale them by their Total Assets
for index, row in df_nodes.iterrows():
    bank = row['bank_id']
    assets = row['total_assets_bln']
    G.add_node(bank, size=assets, label=bank)

# 5. Add the edges (The Exposure)
for index, row in df_edges.iterrows():
    G.add_edge(row['lender'], row['borrower'], weight=row['exposure_bln'])

# 6. Plotting the Graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5, seed=42) 

# Extract sizes and weights for scaling
node_sizes = [nx.get_node_attributes(G, 'size')[node] * 1.5 for node in G.nodes()]
edge_weights = [G[u][v]['weight'] / 15 for u, v in G.edges()]

# Draw the components
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.9, edgecolors='black')
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray', arrows=True, arrowsize=20)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

# Customize the chart aesthetics
plt.title("Transatlantic G-SIB Systemic Credit Exposure Matrix (2023)", fontsize=16, fontweight='bold')
plt.axis("off")
plt.tight_layout()

# Save the high-res image to the root folder
output_path = os.path.join(script_dir, "systemic_network_graph.png")
if 'src' in script_dir:
    output_path = os.path.join(script_dir, '..', "systemic_network_graph.png")

plt.savefig(output_path, dpi=300)
print(f"✅ Saved network graph image to: {output_path}")
plt.show()