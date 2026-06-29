import pandas as pd
import networkx as nx
from pyvis.network import Network
import os

print("🕸️ Initializing INTERACTIVE Systemic Network Topology...")

# --- BULLETPROOF PATH HANDLING ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check if we are in the root folder or the src folder
if os.path.exists(os.path.join(script_dir, 'data')):
    data_dir = os.path.join(script_dir, 'data') # Script is in root
else:
    data_dir = os.path.join(script_dir, '..', 'data') # Script is in src/

edges_path = os.path.join(data_dir, 'network_edges.csv')
nodes_path = os.path.join(data_dir, 'network_nodes.csv')

try:
    df_edges = pd.read_csv(edges_path)
    df_nodes = pd.read_csv(nodes_path)
except FileNotFoundError:
    print(f"❌ Error: Could not find CSV files in {data_dir}.")
    exit()

# 1. Initialize a Directed Graph
G = nx.DiGraph()

# 2. Add the central aggregate node
G.add_node("TOTAL_FINANCIALS", size=50, title="The Global Financial System", label="Financial System", color="#ef4444")

# 3. Add the Bank nodes
for index, row in df_nodes.iterrows():
    bank = row['bank_id']
    assets = row['total_assets_bln']
    
    # Differentiate colors based on region (US vs Euro)
    node_color = "#3b82f6" if assets > 1000 and bank in ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS'] else "#10b981"
    
    # We scale down the size slightly for PyVis so it fits on screen nicely
    scaled_size = max(10, assets / 100) 
    
    # The 'title' attribute creates the hover tooltip!
    hover_text = f"Bank: {bank}\nTotal Assets: ${assets} Billion"
    
    G.add_node(bank, size=scaled_size, title=hover_text, label=bank, color=node_color)

# 4. Add the edges (The Exposure)
for index, row in df_edges.iterrows():
    lender = row['lender']
    borrower = row['borrower']
    exposure = row['exposure_bln']
    
    hover_edge = f"Exposure: ${exposure} Billion"
    G.add_edge(lender, borrower, value=exposure, title=hover_edge, color="#64748b")

# 5. Generate the Interactive HTML Widget
# We turn on the physics engine so the nodes bounce and settle naturally
net = Network(height="800px", width="100%", bgcolor="#0f172a", font_color="white", directed=True)
net.from_nx(G)

# Tweak the physics for a cleaner layout
net.repulsion(node_distance=200, spring_length=200)

output_file = os.path.join(script_dir, "interactive_network.html")
net.save_graph(output_file)

print(f"✅ Saved interactive web graph to: {output_file}")
print("🌐 Double-click that HTML file to open it in your web browser!")