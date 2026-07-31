import pandas as pd
import numpy as np
from pathlib import Path

def ingest_and_clean_data(raw_file_path, processed_dir):
    print("Initializing Production Data Ingestion Pipeline...")
    
    # 1. Path Resolution
    raw_path = Path(raw_file_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"System Error: Could not find dataset at {raw_path.resolve()}")
        
    # Ensure processed directory exists for the pipeline handoff
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
        
    # 2. Ingestion & Cleaning
    # thousands=',' cleanly handles any comma-separated numbers in the CSV
    df = pd.read_csv(raw_path, thousands=',')
    
    # Drop any phantom 'Unnamed' columns created by Excel exports or trailing spaces
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Strip hidden whitespace from column headers just in case
    df.columns = df.columns.str.strip()
    
    print(f"CLEANED COLUMNS DETECTED: {df.columns.tolist()}")
    
    # 3. Data Type Enforcement
    # Forces the exposure column to be numeric so NetworkX math doesn't crash on strings
    if 'Interbank_Exposure_Billion' in df.columns:
        df['Interbank_Exposure_Billion'] = pd.to_numeric(df['Interbank_Exposure_Billion'], errors='coerce')
        
    # 4. Audit Logging
    total_nodes = len(df)
    missing_values = df['Interbank_Exposure_Billion'].isna().sum() if 'Interbank_Exposure_Billion' in df.columns else 0
    
    print("-" * 50)
    print(f"Network Nodes Ingested:        {total_nodes}")
    print(f"Missing/NaN Exposure Values:   {missing_values}")
    print("-" * 50)
    
    # 5. Pipeline Handoff
    # Save the mathematically pure dataset to the processed folder
    processed_file_path = Path(processed_dir) / "cleaned_bank_nodes.csv"
    df.to_csv(processed_file_path, index=False)
    print(f"Pipeline Success: Clean dataset saved to {processed_file_path.resolve()}\n")
    
    return df

# --- Execution ---
if __name__ == "__main__":
    # Raw input path (your manual extraction)
    RAW_FILE = r"E:\study\Github\BlackScholes\global-banking-stress-monitor\data\raw\global_banks_manual_entry.csv"
    
    # Processed output path (routing to the correct folder for Phase 2)
    PROCESSED_DIR = r"E:\study\Github\BlackScholes\global-banking-stress-monitor\data\processed"
    
    # Run the pipeline
    master_df = ingest_and_clean_data(RAW_FILE, PROCESSED_DIR)
    
    # Preview the final matrix ready for NetworkX
    print(master_df.head(15))