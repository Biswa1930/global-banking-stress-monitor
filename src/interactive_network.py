import pandas as pd
import networkx as nx
from pyvis.network import Network
import os

print("🕸️ Initializing INTERACTIVE Systemic Network Topology (39 G-SIBs)...")

# --- PATH HANDLING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')

edges_path = os.path.join(processed_dir, 'network_edges.csv')
centrality_path = os.path.join(processed_dir, 'network_centrality.csv')
names_path = os.path.join(processed_dir, 'bank_names.csv')

try:
    df_edges = pd.read_csv(edges_path)
    df_nodes = pd.read_csv(centrality_path)
    
    # 1. Load the Mapping Dictionary
    if os.path.exists(names_path):
        df_names = pd.read_csv(names_path)
        df_names['ticker'] = df_names['ticker'].str.upper() # Ensure matching
        name_map = dict(zip(df_names['ticker'], df_names['full_name']))
    else:
        name_map = {}
        print("⚠️ Warning: bank_names.csv not found. Using tickers only.")
        
except FileNotFoundError as e:
    print(f"❌ Error: Could not find processed data files.\n{e}")
    exit()

# Region classification
us_banks = ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'BK', 'STT']
eu_banks = ['HSBC', 'BCS', 'DB', 'BNPQY', 'SCGLY', 'UBS', 'ING', 'SAN', 'UNCFF', 'NRDEF', 'BPCE', 'CASA', 'STAN']
asia_banks = ['MUFG', 'SMFG', 'MIZUHO', 'ICBC', 'CCB', 'ABC', 'BOC', 'SBI']

def get_region_color(ticker):
    if ticker in us_banks: return "#3b82f6"
    if ticker in eu_banks: return "#10b981"
    if ticker in asia_banks: return "#8b5cf6"
    return "#f59e0b"

# 1. Initialize a Directed Graph
G = nx.DiGraph()

# 2. Add central node
system_tooltip = "🏦 THE GLOBAL SYSTEM\n========================\nAggregate Counterparty Sink"
G.add_node("SYSTEM", size=50, title=system_tooltip, label="SYSTEM", color="#ef4444")

# 3. Add Bank nodes
all_banks = df_edges['Lender_Ticker'].unique()

for bank in all_banks:
    # 2. Inside the loop: Get full name from mapping
    full_name = name_map.get(bank.upper(), bank)
    
    node_data = df_nodes[df_nodes['Bank_Ticker'] == bank]
    out_exp = node_data['Out_Exposure_Billion'].values[0] if not node_data.empty else 0.0
    pr_score = node_data['PageRank_Score'].values[0] if not node_data.empty else 0.0

    scaled_size = max(15, out_exp / 15) 
    node_color = get_region_color(bank)
    
    # Update hover text to use the full_name variable
    hover_text = (
        f"🏦 INSTITUTION: {full_name}\n"
        f"========================\n"
        f"Systemic Exposure : ${out_exp:,.2f}B\n"
        f"PageRank Score    : {pr_score:.4f}\n"
        f"Data Status       : {'Active' if out_exp > 0 else 'Pending Data'}"
    )
    
    G.add_node(bank, size=scaled_size, title=hover_text, label=bank, color=node_color)

# 1. Update the loading block in src/interactive_network.py
names_path = os.path.join(processed_dir, 'bank_names.csv')

try:
    df_names = pd.read_csv(names_path)
    # Ensure tickers are uppercase to avoid matching bugs
    df_names['ticker'] = df_names['ticker'].str.upper()
    # Create the mapping dictionary
    name_map = dict(zip(df_names['ticker'], df_names['full_name']))
except Exception as e:
    print(f"⚠️ Warning: Could not load bank_names.csv. Using Ticker fallback. Error: {e}")
    name_map = {}

# 2. Inside your loop, it stays the same:
full_name = name_map.get(bank.upper(), bank)

# 4. Add edges
for index, row in df_edges.iterrows():
    lender = row['Lender_Ticker']
    borrower = row['Borrower_Ticker']
    exposure = row['Exposure_Billion_USD']
    visual_weight = max(0.5, exposure / 50)
    edge_tooltip = f"Lending Exposure: ${exposure:,.2f}B" if exposure > 0 else "Exposure: Pending Data"
    G.add_edge(lender, borrower, value=visual_weight, title=edge_tooltip, color="#64748b")

# 5. Generate
net = Network(height="900px", width="100%", bgcolor="#0f172a", font_color="white", directed=True)
net.from_nx(G)
net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=250, spring_strength=0.04, damping=0.09)
net.save_graph(os.path.join(processed_dir, "interactive_network.html"))

print(f"✅ Successfully compiled 39-Bank topology with full names!")