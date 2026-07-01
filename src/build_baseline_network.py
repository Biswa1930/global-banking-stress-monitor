import pandas as pd
import os

print("🕸️ Building Baseline Interbank Network (Phase 1D)...")

# 1. Define Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
reg_file = os.path.join(processed_dir, 'us_banks_regulatory_raw.csv')
out_file = os.path.join(processed_dir, 'network_edges.csv')

# 2. Load the regulatory data we just parsed
df_reg = pd.read_csv(reg_file)

# We only want the most recent year (2023) to establish our current network state
df_2023 = df_reg[df_reg['Year'] == 2023].copy()

# 3. The 39 G-SIBs from your project specification
g_sibs = [
    'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'BK', 'STT',
    'HSBC', 'BCS', 'DB', 'BNPQY', 'SCGLY', 'UBS', 'ING', 'SAN', 'UNCFF', 'NRDEF', 'BPCE', 'CASA', 'STAN', 'GSI',
    'MUFG', 'SMFG', 'MIZUHO',
    'ICBC', 'CCB', 'ABC', 'BOC',
    'CBA', 'ANZ', 'NAB', 'WBC', 'SBI',
    'RY', 'TD', 'BMO', 'BNS', 'CM'
]

# 4. Map the exposures
edges = []
for bank in g_sibs:
    exposure = 0.0
    
    # If it's a US bank, pull the exact 2023 interbank exposure we calculated from the Federal Reserve
    if bank in df_2023['Bank_Ticker'].values:
        exposure = df_2023.loc[df_2023['Bank_Ticker'] == bank, 'Interbank_Exposure_Billion'].values[0]
    
    edges.append({
        'Lender_Ticker': bank,
        'Borrower_Ticker': 'SYSTEM',
        'Exposure_Billion_USD': round(exposure, 2)
    })

# 5. Save the baseline network
df_edges = pd.DataFrame(edges)
df_edges.to_csv(out_file, index=False)

print("\n" + "="*80)
print(f"✅ SUCCESSFULLY BUILT BASELINE NETWORK EDGE LIST")
print("="*80)
print(f"Data saved to: {out_file}")

# Quick verification of JPM
jpm_edge = df_edges[df_edges['Lender_Ticker'] == 'JPM']
print("\n🔍 Verification Check (JPM to SYSTEM Exposure):")
print(jpm_edge.to_string(index=False))