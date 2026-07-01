import pandas as pd
import networkx as nx
from pyvis.network import Network
import os

print("🕸️ Initializing INTERACTIVE Systemic Network Topology (39 G-SIBs)...")

# ==========================================
# 1. BULLETPROOF PATH HANDLING
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')

edges_path = os.path.join(processed_dir, 'network_edges.csv')
centrality_path = os.path.join(processed_dir, 'network_centrality.csv')
names_path = os.path.join(processed_dir, 'ticker_map.csv')

# ==========================================
# 2. DATA INGESTION & MAPPING
# ==========================================
try:
    df_edges = pd.read_csv(edges_path)
    df_nodes = pd.read_csv(centrality_path)
    
    # Safely load the Bank Names mapping dictionary
    if os.path.exists(names_path):
        # 1. Use latin1 to prevent the European accent mark crash
        df_names = pd.read_csv(names_path, encoding='latin1')
        
        # 👇 THE NUCLEAR FIX 👇
        # Ignore Excel's hidden BOM characters completely. 
        # Isolate the first two columns and force them to be exactly what we need.
        df_names = df_names.iloc[:, :2] 
        df_names.columns = ['ticker', 'full_name']
        
        # 3. Force uppercase and strip spaces in the data to prevent matching bugs
        df_names['ticker'] = df_names['ticker'].astype(str).str.strip().str.upper() 
        name_map = dict(zip(df_names['ticker'], df_names['full_name']))
    else:
        name_map = {}
        print("⚠️ Warning: ticker_map.csv not found. Using tickers as fallback.")
        
except FileNotFoundError as e:
    print(f"❌ Error: Could not find core processed data files.\n{e}")
    exit()
except ValueError as e:
    print(f"❌ Error: CSV is not formatted into two columns. Did Excel save it with commas?\n{e}")
    exit()
# ==========================================
# 3. REGION CLASSIFICATION
# ==========================================
us_banks = ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'BK', 'STT']
eu_banks = ['HSBC', 'BCS', 'DB', 'BNPQY', 'SCGLY', 'UBS', 'ING', 'SAN', 'UNCFF', 'NRDEF', 'BPCE', 'CASA', 'STAN']
asia_banks = ['MUFG', 'SMFG', 'MIZUHO', 'ICBC', 'CCB', 'ABC', 'BOC', 'SBI']

def get_region_color(ticker):
    if ticker in us_banks: return "#3b82f6"   # Blue
    if ticker in eu_banks: return "#10b981"   # Green
    if ticker in asia_banks: return "#8b5cf6" # Purple
    return "#f59e0b"                          # Orange

# ==========================================
# 4. BUILD THE GRAPH TOPOLOGY
# ==========================================
G = nx.DiGraph()

# Add the central aggregate node (The System)
system_tooltip = (
    "🏦 THE GLOBAL SYSTEM\n"
    "========================\n"
    "Aggregate Counterparty Sink"
)
G.add_node("SYSTEM", size=50, title=system_tooltip, label="SYSTEM", color="#ef4444")

# Extract all unique banks from the edge list
all_banks = df_edges['Lender_Ticker'].unique()

print(f"📊 Loading {len(all_banks)} bank nodes into the physics engine...")

for bank in all_banks:
    # 1. Resolve Full Name
    full_name = name_map.get(str(bank).upper(), bank)
    
    # 2. Extract Centrality Metrics (Fallback to 0.0 if missing)
    node_data = df_nodes[df_nodes['Bank_Ticker'] == bank]
    if not node_data.empty:
        out_exp = float(node_data['Out_Exposure_Billion'].values[0])
        pr_score = float(node_data['PageRank_Score'].values[0])
    else:
        out_exp = 0.0
        pr_score = 0.0

    # 3. Calculate Visual Attributes
    scaled_size = max(15, out_exp / 15) 
    node_color = get_region_color(bank)
    
    # 4. Generate Clean Plain-Text Tooltip
    hover_text = (
        f"🏦 INSTITUTION: {full_name}\n"
        f"========================\n"
        f"Systemic Exposure : ${out_exp:,.2f}B\n"
        f"PageRank Score    : {pr_score:.4f}\n"
        f"Data Status       : {'Active' if out_exp > 0 else 'Pending Data'}"
    )
    
    # 5. Add to Graph
    G.add_node(bank, size=scaled_size, title=hover_text, label=bank, color=node_color)

# Add the edges (Lending channels)
for index, row in df_edges.iterrows():
    lender = row['Lender_Ticker']
    borrower = row['Borrower_Ticker']
    exposure = float(row['Exposure_Billion_USD'])
    
    visual_weight = max(0.5, exposure / 50)
    edge_tooltip = f"Lending Exposure: ${exposure:,.2f}B" if exposure > 0 else "Exposure: Pending Data"
    
    G.add_edge(lender, borrower, value=visual_weight, title=edge_tooltip, color="#64748b")

# ==========================================
# 5. RENDER INTERACTIVE HTML
# ==========================================
print("⚙️ Compiling HTML and simulating node repulsion...")
net = Network(height="900px", width="100%", bgcolor="#0f172a", font_color="white", directed=True)
net.from_nx(G)

# Professional physics configuration for hub-and-spoke models
net.barnes_hut(
    gravity=-8000,
    central_gravity=0.3,
    spring_length=250,
    spring_strength=0.04,
    damping=0.09
)

# Save output
output_file = os.path.join(processed_dir, "interactive_network.html")
net.save_graph(output_file)

print("\n" + "="*80)
print(f"✅ SUCCESSFULLY EXPORTED SYSTEMIC TOPOLOGY")
print("="*80)
print(f"File saved to: {output_file}")