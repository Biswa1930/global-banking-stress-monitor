
"""
data_pipeline.py
----------------
Global Banking Stress Monitor — Phase 1B Data Ingestion
Fetches macro stress indicators (FRED) and G-SIB equity microstructure (yfinance via OpenBB).

Usage:
    python src/data_pipeline.py

Requirements:
    - .env file in project root with FRED_API_KEY=your_key
    - pip install fredapi openbb python-dotenv pandas numpy

Output:
    data/raw/market_data/macro_stress_indicators.csv
    data/raw/market_data/gsib_log_returns.csv
    data/raw/market_data/gsib_prices_raw.csv
    data/raw/market_data/data_coverage_report.csv
"""

from fredapi import Fred
from openbb import obb
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError(
        "FRED_API_KEY not found.\n"
        "Create a .env file in your project root with:\n"
        "  FRED_API_KEY=your_key_here\n"
        "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
    )

# ── Constants ─────────────────────────────────────────────────────────────────
START_DATE = "2019-01-01"
END_DATE   = "2024-01-01"
OUTPUT_DIR = os.path.join("data", "raw", "market_data")

# ── The official FSB G-SIB Universe (50 Banks) ───────────────────────────────
# Using US / ADR tickers for timezone alignment.
# OTC tickers (suffix F/Y) have thinner volume — flag in coverage report.
GSIB_TICKERS = [
    # United States (8)
    "JPM", "BAC", "C", "WFC", "GS", "MS", "BK", "STT",
    # Europe (14)
    "HSBC", "BCS", "DB", "UBS", "ING", "SAN",
    "CS",                                               # Credit Suisse — delisted Jun 2023, handled below
    "BNPQY", "SCGLY", "CRARY", "SCBFF",
    "UNCFF", "NRDBY", "DNKEY",
    # Japan (3)
    "MUFG", "SMFG", "MFG",
    # China (4)
    "IDCBY", "CICHY", "ACGBY", "BACHY",
    # Other Asia / Australia (6)
    "DBSDY", "HDB", "CMWAY", "WEBNF", "ANZGY", "NABZY",
    # Canada (5)
    "RY", "TD", "BNS", "BMO", "CM",
]

# Banks with known corporate actions that corrupt post-event data.
# Data is set to NaN from the delist date onward — NOT dropped —
# so the full panel stays rectangular and the gap is documented.
DELISTED = {
    "CS": "2023-06-12",   # UBS acquisition of Credit Suisse completed
}


# ── 1. FRED macro stress indicators ──────────────────────────────────────────
def fetch_macro_fred(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull 5 macro stress indicators from FRED.

    Series used:
        VIXCLS       — CBOE Volatility Index (daily)
        USD3MTD156N  — 3-Month USD LIBOR / SOFR transition rate (daily)
        DTB3         — 3-Month T-Bill secondary market rate (daily)
        BAMLH0A0HYM2 — ICE BofA US High Yield Option-Adjusted Spread (daily)
        T10Y2Y       — 10-Year minus 2-Year Treasury yield spread (daily)

    TED spread is computed manually as USD3MTD156N minus DTB3.
    TEDRATE was discontinued by FRED in 2023 and is NOT used.

    Returns:
        pd.DataFrame with columns:
            VIX_INDEX, TED_SPREAD, FRA_OIS_SPREAD,
            HY_CREDIT_SPREAD, YIELD_CURVE_2Y10Y
    """
    print("\n" + "="*60)
    print("FRED PIPELINE — Macro Stress Indicators")
    print(f"Period: {start} → {end}")
    print("="*60)

    fred = Fred(api_key=api_key)

    # Series to pull directly
    direct_series = {
        "VIX_INDEX":        "VIXCLS",
        "HY_CREDIT_SPREAD": "BAMLH0A0HYM2",
        "YIELD_CURVE_2Y10Y":"T10Y2Y",
        # Helper series for TED computation — not kept in final output
        # USD3MTD156N (LIBOR) discontinued Jun 2023 — replaced with SOFR
        "_SOFR":            "SOFR",
        "_TBILL_3M":        "DTB3",
    }

    # Fetch each series individually and ffill immediately before joining.
    # Critical: SOFR, DTB3, VIX, and bond spread series have slightly different
    # publication calendars on FRED. Joining first then calling ffill().dropna()
    # causes cross-series NaN misalignment that silently drops years of data
    # (observed: 2019-2024 window collapsed to 141 rows without this fix).
    series_dict = {}

    for name, series_id in direct_series.items():
        print(f"  → Downloading {name} ({series_id})")
        try:
            s = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            series_dict[name] = s.ffill()   # ffill each series independently
        except Exception as e:
            print(f"    [!] Error fetching {name}: {e}")

    # Build a unified business-day date index, reindex all series to it
    all_dates = pd.date_range(start=start, end=end, freq="B")
    macro_df  = pd.DataFrame(index=all_dates)
    for name, series in series_dict.items():
        macro_df[name] = series.reindex(all_dates, method="ffill")

    # Compute TED spread manually using SOFR (LIBOR successor, post-Jun 2023)
    # SOFR available from Apr 2018 onward — covers our full 2019-2024 window
    if "_SOFR" in macro_df.columns and "_TBILL_3M" in macro_df.columns:
        macro_df["TED_SPREAD"]     = macro_df["_SOFR"] - macro_df["_TBILL_3M"]
        macro_df["FRA_OIS_SPREAD"] = macro_df["_SOFR"]   # SOFR as FRA-OIS proxy
        macro_df.drop(columns=["_SOFR", "_TBILL_3M"], inplace=True)
        print("  → TED_SPREAD computed: SOFR minus DTB3 (3M T-Bill)")
        print("  → FRA_OIS_SPREAD proxy: SOFR (LIBOR successor, available from Apr 2018)")
        print("  → Note: SOFR is overnight; pre-2023 LIBOR was 3M term — document in data_notes.md")
    else:
        print("  [!] Could not compute TED_SPREAD — helper series missing")

    # Drop only rows where ALL columns are NaN (before any series starts)
    macro_df = macro_df.dropna(how="all")

    print(f"\n  ✓ Macro data shape   : {macro_df.shape}")
    print(f"  ✓ Date range         : {macro_df.index[0].date()} → {macro_df.index[-1].date()}")
    print(f"  ✓ Columns            : {list(macro_df.columns)}")

    return macro_df


# ── 2. G-SIB equity microstructure ───────────────────────────────────────────
def fetch_bank_microstructure(
    tickers: list,
    start: str,
    end: str,
    delisted: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull daily closing prices for all G-SIB tickers via OpenBB / yfinance.
    Computes log returns. Handles delistings explicitly.

    Args:
        tickers  : list of ticker strings
        start    : start date string YYYY-MM-DD
        end      : end date string YYYY-MM-DD
        delisted : dict {ticker: delist_date_string} for corporate action handling

    Returns:
        Tuple of (log_returns_df, raw_prices_df)
        Both are indexed by date with tickers as columns.
    """
    print("\n" + "="*60)
    print(f"MARKET PIPELINE — G-SIB Equity Microstructure ({len(tickers)} banks)")
    print(f"Period: {start} → {end}")
    print("="*60)

    prices_df  = pd.DataFrame()
    failed     = []

    for ticker in tickers:
        try:
            print(f"  → Fetching {ticker}")
            data = obb.equity.price.historical(
                symbol=ticker,
                start_date=start,
                end_date=end,
                provider="yfinance",
            ).to_df()
            prices_df[ticker] = data["close"]
        except Exception as e:
            print(f"    [!] Failed: {ticker} — {e}")
            failed.append(ticker)

    if failed:
        print(f"\n  [WARNING] {len(failed)} tickers failed entirely: {failed}")
        print("  Check tickers, OTC liquidity, or yfinance availability.")

    # ── Handle known delistings ───────────────────────────────────────────────
    print("\n  Applying corporate action adjustments...")
    for ticker, delist_date in delisted.items():
        if ticker in prices_df.columns:
            n_nulled = prices_df.loc[delist_date:, ticker].notna().sum()
            prices_df.loc[delist_date:, ticker] = np.nan
            print(f"    → {ticker}: set {n_nulled} rows to NaN from {delist_date} (delisted/acquired)")

    # ── Data coverage report ──────────────────────────────────────────────────
    coverage = (prices_df.notna().sum() / len(prices_df) * 100).round(1)
    poor     = coverage[coverage < 80]
    if not poor.empty:
        print(f"\n  [WARNING] {len(poor)} tickers with <80% data coverage:")
        for t, pct in poor.items():
            print(f"    {t}: {pct}%")
        print("  These tickers are OTC ADRs with thin data — document in data_notes.md")

    # Normalise index — OpenBB returns datetime.date objects, not datetime.datetime.
    # pd.to_datetime() ensures .date() calls and downstream merges work correctly.
    prices_df.index = pd.to_datetime(prices_df.index)

    print(f"\n  ✓ Prices shape       : {prices_df.shape}")
    print(f"  ✓ Date range         : {prices_df.index[0].date()} → {prices_df.index[-1].date()}")
    print(f"  ✓ Avg coverage       : {coverage.mean():.1f}%")

    # ── Log returns ───────────────────────────────────────────────────────────
    # thresh: keep rows where at least 80% of tickers have valid data.
    # This handles holidays/half-days without contaminating the matrix.
    log_returns = np.log(prices_df / prices_df.shift(1))
    threshold   = int(len(prices_df.columns) * 0.80)
    log_returns = log_returns.dropna(thresh=threshold)

    print(f"  ✓ Log returns shape  : {log_returns.shape}")
    print(f"    (rows dropped vs raw: {len(prices_df) - 1 - len(log_returns)})")

    return log_returns, prices_df


# ── 3. Coverage report ────────────────────────────────────────────────────────
def build_coverage_report(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce a per-ticker data quality summary saved to CSV.
    Flags tickers that need attention in data_notes.md.
    """
    coverage = prices_df.notna().sum() / len(prices_df) * 100
    first_valid = prices_df.apply(lambda c: c.first_valid_index())
    last_valid  = prices_df.apply(lambda c: c.last_valid_index())

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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Run pipelines ─────────────────────────────────────────────────────────
    macro_data             = fetch_macro_fred(FRED_API_KEY, START_DATE, END_DATE)
    bank_returns, raw_prices = fetch_bank_microstructure(
        GSIB_TICKERS, START_DATE, END_DATE, DELISTED
    )
    coverage_report        = build_coverage_report(raw_prices, macro_data)

    # ── Save outputs ──────────────────────────────────────────────────────────
    macro_path    = os.path.join(OUTPUT_DIR, "macro_stress_indicators.csv")
    returns_path  = os.path.join(OUTPUT_DIR, "gsib_log_returns.csv")
    prices_path   = os.path.join(OUTPUT_DIR, "gsib_prices_raw.csv")
    coverage_path = os.path.join(OUTPUT_DIR, "data_coverage_report.csv")

    macro_data.to_csv(macro_path)
    bank_returns.to_csv(returns_path)
    raw_prices.to_csv(prices_path)
    coverage_report.to_csv(coverage_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1B COMPLETE — Data Ingestion Summary")
    print("="*60)
    print(f"  macro_stress_indicators.csv  : {macro_data.shape}")
    print(f"  gsib_log_returns.csv         : {bank_returns.shape}")
    print(f"  gsib_prices_raw.csv          : {raw_prices.shape}")
    print(f"  data_coverage_report.csv     : {coverage_report.shape}")
    print(f"\n  Tickers flagged OK           : {(coverage_report['flag']=='OK').sum()}")
    print(f"  Tickers flagged THIN_DATA    : {(coverage_report['flag']=='THIN_DATA').sum()}")
    print(f"  Tickers flagged DELISTED     : {(coverage_report['flag']=='DELISTED').sum()}")
    print("\n  Next step → src/network_analysis.py")
    print("="*60)