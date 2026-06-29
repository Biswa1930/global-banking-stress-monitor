"""
Data Ingestion Pipeline - Global Banking Stress Monitor
-------------------------------------------------------
Fetches macroeconomic stress indicators (FRED) and bank microstructure
equity data (Yahoo Finance). Handles missing data, delisted tickers, 
and computes standard log returns for G-SIB entities.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Tuple, List
import warnings

# Suppress yfinance timezone and pandas future warnings for clean console output
warnings.filterwarnings("ignore")

def fetch_macro_fred(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches macroeconomic systemic risk indicators from FRED via REST API.
    
    Args:
        api_key (str): FRED API key.
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        
    Returns:
        pd.DataFrame: Cleaned, forward-filled macro time-series data.
    """
    print("📥 Fetching macro stress indicators from FRED...")
    
    # Core macro-prudential risk indicators
    series_dict = {
        'VIX': 'VIXCLS',                  # Market Volatility
        'TED_Spread': 'TEDRATE',          # Interbank Trust/Liquidity
        'Yield_Curve_2y10y': 'T10Y2Y',    # Recession Indicator
        'High_Yield_Spread': 'BAMLH0A0HYM2' # Corporate Credit Risk
    }

    macro_df = pd.DataFrame()

    if api_key == "YOUR_API_KEY_HERE" or not api_key:
        print("⚠️ WARNING: Default or missing FRED API key. Skipping macro fetch.")
        return pd.DataFrame(columns=series_dict.keys())

    try:
        for name, series_id in series_dict.items():
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json().get('observations', [])
                temp_df = pd.DataFrame(data)[['date', 'value']]
                temp_df['value'] = pd.to_numeric(temp_df['value'], errors='coerce')
                temp_df.set_index('date', inplace=True)
                temp_df.rename(columns={'value': name}, inplace=True)

                if macro_df.empty:
                    macro_df = temp_df
                else:
                    macro_df = macro_df.join(temp_df, how='outer')

        # Forward fill weekends and bank holidays, then drop remaining NaNs
        macro_df = macro_df.ffill().dropna(how='all')
        print("✅ Macro data fetched successfully.")
        return macro_df

    except Exception as e:
        print(f"❌ Critical Error fetching macro data: {e}")
        return pd.DataFrame()


def fetch_bank_microstructure(tickers: List[str], start_date: str, end_date: str, delisted: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches daily equity prices and calculates quantitative log returns.
    """
    if delisted is None:
        delisted = []

    active_tickers = [t for t in tickers if t not in delisted]
    print(f"📥 Fetching microstructure data for {len(active_tickers)} active G-SIBs...")

    try:
        # FIX: auto_adjust=True automatically accounts for splits/dividends 
        # and stores it under the 'Close' column, making it immune to yfinance version changes.
        df_raw = yf.download(active_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # Safely extract the prices depending on yfinance's response structure
        if 'Close' in df_raw.columns:
            df_prices = df_raw['Close']
        elif 'Adj Close' in df_raw.columns:
            df_prices = df_raw['Adj Close']
        else:
            raise ValueError(f"Unexpected yfinance columns: {df_raw.columns}")

        # Ensure single-ticker fetches remain DataFrames, not Series
        if isinstance(df_prices, pd.Series):
            df_prices = df_prices.to_frame(name=active_tickers[0])

        # Calculate continuous log returns: ln(P_t / P_{t-1})
        df_returns = np.log(df_prices / df_prices.shift(1)).dropna(how='all')

        print("✅ Microstructure data fetched successfully.")
        return df_returns, df_prices

    except Exception as e:
        print(f"❌ Critical Error fetching microstructure data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def build_coverage_report(prices_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a data quality report flagging thin data or delisted entities.
    Crucial for identifying structural breaks in historical backtesting.
    """
    if prices_df.empty:
        return pd.DataFrame()

    coverage = prices_df.notna().mean() * 100
    first_valid = prices_df.apply(lambda x: x.first_valid_index())
    last_valid = prices_df.apply(lambda x: x.last_valid_index())
    
    report = pd.DataFrame({
        "coverage_pct": coverage.round(1),
        "first_valid":  first_valid,
        "last_valid":   last_valid,
        "total_rows":   len(prices_df),
        "valid_rows":   prices_df.notna().sum(),
        "flag":         coverage.apply(
            lambda x: "DELISTED"   if x < 60
                 else "THIN_DATA"  if x < 80
                 else "OK"
        ),
    })
    
    return report.sort_values("coverage_pct")