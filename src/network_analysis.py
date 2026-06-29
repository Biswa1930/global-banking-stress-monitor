"""
Network Topology Engine - Global Banking Stress Monitor
-------------------------------------------------------
Constructs directed interbank exposure networks and computes 
systemic importance via eigenvalue and betweenness centrality.
"""

import pandas as pd
import networkx as nx
from typing import Tuple

def build_interbank_network(edge_df: pd.DataFrame) -> nx.DiGraph:
    """
    Builds a directed, weighted NetworkX graph from an edge list.
    
    Args:
        edge_df (pd.DataFrame): Must contain 'lender', 'borrower', and 'exposure' columns.
        
    Returns:
        nx.DiGraph: Directed graph representing contagion channels.
    """
    print("🕸️ Constructing directed interbank exposure network...")
    
    G = nx.DiGraph()
    
    # Add weighted edges (Exposure in USD Billions)
    for _, row in edge_df.iterrows():
        G.add_edge(
            row['lender'], 
            row['borrower'], 
            weight=row['exposure']
        )
        
    print(f"✅ Network constructed: {G.number_of_nodes()} banks, {G.number_of_edges()} exposure links.")
    return G

def compute_centrality_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Calculates key systemic risk topology metrics.
    
    Args:
        G (nx.DiGraph): The interbank network.
        
    Returns:
        pd.DataFrame: Table of centrality scores per bank.
    """
    print("🧮 Computing eigenvector and betweenness centralities...")
    
    # 1. Eigenvector Centrality (Who is connected to other highly connected banks?)
    # Max iterations increased for complex global networks
    eigen_cent = nx.eigenvector_centrality_numpy(G, weight='weight')
    
    # 2. Betweenness Centrality (Who acts as a critical bridge/bottleneck?)
    between_cent = nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    # 3. Out-Degree Infectivity (Total outward exposure)
    out_degree = dict(G.out_degree(weight='weight'))
    
    # 4. In-Degree Vulnerability (Total inward reliance)
    in_degree = dict(G.in_degree(weight='weight'))

    # Compile into a clean matrix
    metrics_df = pd.DataFrame({
        'Bank': list(G.nodes()),
        'Eigenvector_Systemic_Importance': [eigen_cent.get(n, 0) for n in G.nodes()],
        'Betweenness_Highway_Score': [between_cent.get(n, 0) for n in G.nodes()],
        'Out_Degree_Infectivity': [out_degree.get(n, 0) for n in G.nodes()],
        'In_Degree_Vulnerability': [in_degree.get(n, 0) for n in G.nodes()]
    }).set_index('Bank')
    
    return metrics_df.sort_values(by='Eigenvector_Systemic_Importance', ascending=False)