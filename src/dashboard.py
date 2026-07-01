import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. Page Configuration
st.set_page_config(page_title="Global Banking Stress Monitor", page_icon="🏦", layout="wide")
st.title("🏦 Global Banking Stress Monitor")
st.markdown("### Quantitative Systemic Risk Dashboard")
st.markdown(r"An institutional-grade risk engine aggregating Network Centrality, $\Delta$CoVaR, and SRISK metrics.")

# 2. Data Loading Engine
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    srisk = pd.read_csv(os.path.join(processed_dir, 'srisk_results.csv'))
    covar = pd.read_csv(os.path.join(processed_dir, 'covar_results.csv'))
    absorption = pd.read_csv(os.path.join(processed_dir, 'absorption_ratio.csv'))
    network = pd.read_csv(os.path.join(processed_dir, 'network_centrality.csv'))
    
    # Clean up dates for the absorption ratio
    absorption['Date'] = pd.to_datetime(absorption['Date'])
    return srisk, covar, absorption, network

try:
    df_srisk, df_covar, df_abs, df_net = load_data()
except Exception as e:
    st.error(f"❌ Missing data files. Ensure Phase 1 and 2 scripts have successfully executed. Error: {e}")
    st.stop()

# 3. Dashboard Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ SRISK (Capital Vulnerability)", 
    "📉 ΔCoVaR (Systemic Spillovers)", 
    "🌀 Systemic Fragility (PCA)",
    "🕸️ Network Topology"
])

# --- TAB 1: SRISK ---
with tab1:
    st.subheader("Expected Capital Shortfall Under Severe Market Stress")
    st.markdown("Calculated via Brownlees & Engle (2017) methodology. Represents the expected capital depletion during a 40% market collapse.")
    
    fig_srisk = px.bar(
        df_srisk, 
        x='Bank_Ticker', 
        y='SRISK_Billion',
        color='SRISK_Billion', 
        color_continuous_scale='Reds',
        labels={'SRISK_Billion': 'Capital Shortfall ($B)'},
        title="SRISK by Institution (Billions USD)"
    )
    st.plotly_chart(fig_srisk, width="stretch")
    
    st.dataframe(df_srisk.style.format({
        'MES_Daily': '{:.4f}', 'LRMES_6Mo': '{:.4f}', 
        'Equity_Billion': '${:.2f}B', 'Debt_Billion': '${:.2f}B', 'SRISK_Billion': '${:.2f}B'
    }))

# --- TAB 2: ΔCoVaR ---
with tab2:
    st.subheader(r"Marginal Systemic Risk Contribution ($\Delta$CoVaR)")
    st.markdown("Calculated via Adrian & Brunnermeier (2016) conditional quantile regression. Measures the widening of systemic Value-at-Risk when the target institution enters distress.")
    
    # Sort so the most negative (most dangerous) is first
    df_covar_sorted = df_covar.sort_values('Delta_CoVaR')
    
    fig_covar = px.bar(
        df_covar_sorted, 
        x='Bank_Ticker', 
        y='Delta_CoVaR',
        color='Delta_CoVaR', 
        color_continuous_scale='Reds_r',
        labels={'Delta_CoVaR': r'$\Delta$CoVaR (More negative = Worse)'},
        title=r"Systemic Spillover Risk ($\Delta$CoVaR)"
    )
    st.plotly_chart(fig_covar, width="stretch")
    
    st.dataframe(df_covar_sorted.style.format({
        'VaR_5pct': '{:.4f}', 'Beta_Distress': '{:.4f}', 'Delta_CoVaR': '{:.6f}'
    }))

# --- TAB 3: ABSORPTION RATIO ---
with tab3:
    st.subheader("Market Absorption Ratio (Systemic Fragility)")
    st.markdown("Variance explained by the 1st Principal Component (252-day rolling PCA). A ratio spiking above 0.70 indicates high systemic coupling and fragility.")
    
    fig_abs = px.line(
        df_abs, 
        x='Date', 
        y='Absorption_Ratio',
        title="Cross-Sectional Variance Explained by Systemic Factor",
        labels={'Absorption_Ratio': 'Absorption Ratio (1st PC)'}
    )
    
    # Add a critical threshold line
    fig_abs.add_hline(y=0.70, line_dash="dash", line_color="red", annotation_text="Danger Threshold (70%)")
    st.plotly_chart(fig_abs, width="stretch")

# --- TAB 4: NETWORK CENTRALITY ---
with tab4:
    st.subheader("Interbank Exposure Topology")
    st.markdown("Topological risk metrics derived from Federal Reserve FR Y-9C filings and Pillar 3 disclosures.")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_out = px.bar(
            df_net.sort_values('Out_Exposure_Billion', ascending=False), 
            x='Bank_Ticker', 
            y='Out_Exposure_Billion',
            title="Total Systemic Outward Exposure ($B)"
        )
        st.plotly_chart(fig_out, width="stretch")
        
    with col2:
        fig_pr = px.bar(
            df_net.sort_values('PageRank_Score', ascending=False), 
            x='Bank_Ticker', 
            y='PageRank_Score',
            title="Systemic Importance (PageRank)",
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_pr, width="stretch")