import pandas as pd
import zipfile
import os
import csv

print("🏦 Initializing Final FR Y-9C Regulatory Data Parser...")

# 1. Define Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'raw', 'fed_fry9c')
bulk_dir = os.path.join(data_dir, 'bulk')
output_csv = os.path.join(script_dir, '..', 'data', 'processed', 'us_banks_regulatory_raw.csv')

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# 2. Target US G-SIBs mapping (RSSD ID -> Ticker)
rssd_to_ticker = {
    1039502: 'JPM',
    1073757: 'BAC',
    1951350: 'C',
    1120754: 'WFC',
    2380443: 'GS',
    2162966: 'MS',  # <-- THE FIX: Corrected Morgan Stanley RSSD ID
    3587146: 'BK',
    1111435: 'STT'
}

results = []
print("📂 Scanning for ZIP archives...")

# 3. Process each year's Q4 filing
for year in range(2019, 2024):
    possible_paths = [
        os.path.join(bulk_dir, f"BHCF{year}Q4.zip"),
        os.path.join(data_dir, f"BHCF{year}1231.ZIP"),
        os.path.join(data_dir, f"BHCF{year}1231.zip"),
        os.path.join(bulk_dir, f"BHCF{year}1231.ZIP")
    ]
    
    zip_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not zip_path:
        continue

    print(f"🔄 Extracting and parsing {year} data...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        data_files = [f for f in z.namelist() if f.upper().endswith('.TXT') or f.upper().endswith('.CSV')]
        target_file = data_files[0]
        
        with z.open(target_file) as f:
            first_line = f.readline().decode('utf-8', errors='ignore')
            f.seek(0)
            sep = '^' if '^' in first_line else '\t' if '\t' in first_line else ','
            df = pd.read_csv(f, sep=sep, low_memory=False, encoding='utf-8', on_bad_lines='skip', quoting=csv.QUOTE_NONE)

    df.columns = [str(c).upper().strip() for c in df.columns]
    id_col = 'RSSD9001' if 'RSSD9001' in df.columns else 'IDRSSD' if 'IDRSSD' in df.columns else None
    
    if not id_col:
        continue

    df_banks = df[df[id_col].isin(rssd_to_ticker.keys())]

# 4. Extract and Calculate Fields
    for _, row in df_banks.iterrows():
        rssd = int(row[id_col])
        ticker = rssd_to_ticker[rssd]

        # Hyper-Targeted Suffix Extractor
        def get_val(suffixes):
            for suf in suffixes:
                for col in row.index:
                    col_str = str(col).strip()
                    if col_str.endswith(suf) and pd.notna(row[col]):
                        try:
                            return float(row[col])
                        except ValueError:
                            pass
            return 0.0

        # Hyper-Targeted Suffix Aggregator
        def get_total(suffixes):
            total = 0.0
            for suf in suffixes:
                for col in row.index:
                    col_str = str(col).strip()
                    if col_str.endswith(suf) and pd.notna(row[col]):
                        try:
                            total += float(row[col])
                        except ValueError:
                            pass
                        break # Prevent double-counting if multiple columns match
            return total

        # Assets
        assets_b = get_val(['2170']) / 1_000_000
        

# Capital: P793 is Standardized CET1, WP793 is Advanced CET1
        cet1_std = get_val(['P793']) 
        cet1_adv = get_val(['WP793'])
        
        # Interbank: Fed Funds Sold (B987) + Reverse Repos (B989)
        interbank_b = get_total(['B987', 'B989']) / 1_000_000

        # Derivatives: Gross Trading (A126) + Gross Non-Trading (A127)
        deriv_raw = get_total(['A126', 'A127'])
        if deriv_raw == 0: # Legacy fallback for older years
            deriv_raw = get_total(['3450', '3451', '8693', '8694', '8695'])
        deriv_t = deriv_raw / 1_000_000_000

        results.append({
            'Bank_Ticker': ticker,
            'Year': year,
            'Quarter': 'Q4',
            'Total_Assets_Billion': round(assets_b, 2),
            'CET1_Ratio_Standardized': round(cet1_std, 2),
            'CET1_Ratio_Advanced': round(cet1_adv, 2),
            'Interbank_Exposure_Billion': round(interbank_b, 2),
            'Derivative_Notional_Trillion': round(deriv_t, 2)
        })
# 5. Save the compiled dataset
df_final = pd.DataFrame(results)

if not df_final.empty:
    df_final = df_final.sort_values(by=['Year', 'Total_Assets_Billion'], ascending=[False, False])
    df_final.to_csv(output_csv, index=False)
    
    print("\n" + "="*80)
    print(f"✅ SUCCESSFULLY COMPILED REGULATORY DATA (Found {len(df_final)} records)")
    print("="*80)
    print(f"Data saved to: {output_csv}")
    
    jpm_check = df_final[(df_final['Bank_Ticker'] == 'JPM') & (df_final['Year'] == 2023)]
    if not jpm_check.empty:
        print("\n🔍 Verification Check (JPM 2023):")
        print(jpm_check.to_string(index=False))
else:
    print("❌ No data was extracted.")