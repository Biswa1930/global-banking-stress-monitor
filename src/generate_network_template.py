import pandas as pd
import os

print("🕸️ Initializing Phase 1D: Network Edge Template Generator...")

script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)
output_file = os.path.join(processed_dir, 'network_edges_template.csv')

# The 39 G-SIBs from your project specification
g_sibs = [
    # US (8)
    'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'BK', 'STT',
    # Europe (14)
    'HSBC', 'BCS', 'DB', 'BNPQY', 'SCGLY', 'UBS', 'ING', 'SAN', 'UNCFF', 'NRDEF', 'BPCE', 'CASA', 'STAN', 'GSI',
    # Japan (3)
    'MUFG', 'SMFG', 'MIZUHO',
    # China (4)
    'ICBC', 'CCB', 'ABC', 'BOC',
    # APAC / Other (5)
    'CBA', 'ANZ', 'NAB', 'WBC', 'SBI',
    # Canada (5)
    'RY', 'TD', 'BMO', 'BNS', 'CM'
]

# We will create a template where every bank is a potential lender to a central "SYSTEM" node, 
# and you can manually add specific bilateral (Bank-to-Bank) exposures as you find them in the 10-Ks.
data = []
for bank in g_sibs:
    data.append({
        'Lender_Ticker': bank,
        'Borrower_Ticker': 'SYSTEM', # Placeholder for general aggregate exposure
        'Exposure_Billion_USD': ''   # To be filled manually
    })

df_template = pd.DataFrame(data)
df_template.to_csv(output_file, index=False)

print(f"✅ Template successfully generated: {output_file}")
print("You can now open this CSV in Excel and begin your manual data entry from the annual reports.")