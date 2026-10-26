"""
Assignment II - Graph Analytics for E-Commerce Fraud Detection & Fraud Ring Discovery
Course: COMP8035041 - Big Data Analytics
Student: Kevin Sebastian (NIM: 2702746012)
BINUS Graduate Program - Magister Teknik Informatika

This script executes:
1. Heterogeneous Graph Construction (Transactions, IP Subnets /16, Customer Locations, Delivery States)
2. Projected Transaction Risk Network / Fraud Ring Network
3. Graph Topology Metrics (Nodes, Edges, Density, Connected Components, Modularity)
4. Centrality Analysis (Degree Centrality, PageRank, Betweenness Centrality)
5. Community Detection (Greedy Modularity / Louvain) & Community Fraud Rate Profiling
6. Generation of Publication-Quality Network Visualizations in diagrams/
7. Export of numerical results to JSON and TXT
"""

import os
import sys
import time
import json
import re
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity
import matplotlib.pyplot as plt
import seaborn as sns

# Styling for publication-quality charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def parse_state(addr):
    """Extract 2-letter state code from unstructured address string."""
    if not isinstance(addr, str) or not addr.strip():
        return 'UNK'
    lines = addr.strip().split('\n')
    last_line = lines[-1] if len(lines) > 1 else lines[0]
    m = re.search(r',\s*([A-Z]{2})\s+\d{5}', last_line)
    if m:
        return m.group(1)
    m_st = re.search(r'\b([A-Z]{2})\b', last_line)
    return m_st.group(1) if m_st else 'UNK'

def run_graph_analytics():
    os.makedirs('diagrams', exist_ok=True)
    print("================================================================================")
    print("      ASSIGNMENT II - GRAPH ANALYTICS & FRAUD RING DISCOVERY EXECUTION")
    print("================================================================================")

    v2_path = 'dataset/Fraudulent_E-Commerce_Transaction_Data_2.csv'
    print(f"[*] Step 1: Loading Dataset from {v2_path}...")
    df = pd.read_csv(v2_path)
    total_tx = len(df)
    total_fraud = int(df['Is Fraudulent'].sum())
    print(f"    Loaded {total_tx:,} transactions | Fraudulent: {total_fraud:,} ({total_fraud/total_tx*100:.2f}%)")

    # Step 2: Extract Graph Entities
    print("[*] Step 2: Extracting Multi-Entity Graph Components...")
    # IP Subnet /16 (B-class network prefix representing ISP/proxy block)
    df['ip_subnet_16'] = df['IP Address'].apply(lambda x: '.'.join(str(x).split('.')[:2]) if pd.notnull(x) else 'UNK')
    # Shipping state
    df['shipping_state'] = df['Shipping Address'].apply(parse_state)
    # Binary risk indicators
    df['is_high_amount'] = (df['Transaction Amount'] > 1000.0).astype(int)
    df['is_dawn_tx'] = ((df['Transaction Hour'] >= 0) & (df['Transaction Hour'] <= 5)).astype(int)
    df['is_new_account'] = (df['Account Age Days'] <= 30).astype(int)

    # 1. Full Heterogeneous Graph Construction
    print("[*] Step 3: Constructing Heterogeneous Entity-Sharing Graph...")
    H = nx.Graph()
    for _, r in df.iterrows():
        tx_id = f"TX_{str(r['Transaction ID'])[:8]}"
        sub_id = f"SUB_{r['ip_subnet_16']}"
        loc_id = f"LOC_{r['Customer Location']}"
        st_id = f"ST_{r['shipping_state']}"

        H.add_node(tx_id, ntype='transaction', is_fraud=int(r['Is Fraudulent']), amount=float(r['Transaction Amount']))
        H.add_node(sub_id, ntype='subnet', is_fraud=0)
        H.add_node(loc_id, ntype='location', is_fraud=0)
        H.add_node(st_id, ntype='state', is_fraud=0)

        H.add_edge(tx_id, sub_id, relation='uses_subnet')
        H.add_edge(tx_id, loc_id, relation='originates_location')
        H.add_edge(tx_id, st_id, relation='ships_to_state')

    total_h_nodes = H.number_of_nodes()
    total_h_edges = H.number_of_edges()
    print(f"    - Full Heterogeneous Graph: {total_h_nodes:,} nodes, {total_h_edges:,} edges")

    # 2. Focused Transaction Risk Network (Projected Graph on Fraud & High-Risk Entities)
    # In fraud investigation, security operations center (SOC) analysts project bipartite graphs
    # into a Transaction Risk Graph connecting transactions that share suspicious infrastructure.
    print("[*] Step 4: Constructing Projected Transaction Risk Network...")
    
    # Subnets and locations with known fraud
    fraud_subnets = set(df[df['Is Fraudulent'] == 1]['ip_subnet_16'])
    fraud_locations = set(df[df['Is Fraudulent'] == 1]['Customer Location'])

    # Filter transactions that touch fraud subnets or locations
    risk_tx_df = df[df['ip_subnet_16'].isin(fraud_subnets) | df['Customer Location'].isin(fraud_locations)].copy()
    risk_tx_count = len(risk_tx_df)
    fraud_in_risk = int(risk_tx_df['Is Fraudulent'].sum())
    print(f"    - Filtered Risk Cohort: {risk_tx_count:,} transactions containing {fraud_in_risk:,} frauds ({fraud_in_risk/risk_tx_count*100:.2f}% fraud rate)")

    # Build Transaction Risk Network G
    G = nx.Graph()
    # To maintain optimal visualization and exact betweenness computation, take representative sample if large
    sample_risk = risk_tx_df.head(1500) if len(risk_tx_df) > 1500 else risk_tx_df

    # Add transaction nodes
    for _, r in sample_risk.iterrows():
        tx_id = f"TX_{str(r['Transaction ID'])[:8]}"
        G.add_node(
            tx_id,
            ntype='transaction',
            is_fraud=int(r['Is Fraudulent']),
            amount=float(r['Transaction Amount']),
            subnet=r['ip_subnet_16'],
            location=r['Customer Location'],
            hour=int(r['Transaction Hour'])
        )

    # Add edges between transactions sharing IP Subnet or Location
    # Group by subnet
    for sub, group in sample_risk.groupby('ip_subnet_16'):
        if len(group) > 1:
            tx_list = [f"TX_{str(t)[:8]}" for t in group['Transaction ID']]
            for i in range(len(tx_list)):
                for j in range(i + 1, len(tx_list)):
                    if G.has_edge(tx_list[i], tx_list[j]):
                        G[tx_list[i]][tx_list[j]]['weight'] += 2.0
                        G[tx_list[i]][tx_list[j]]['reasons'].append('shared_subnet')
                    else:
                        G.add_edge(tx_list[i], tx_list[j], weight=2.0, reasons=['shared_subnet'])

    # Group by location
    for loc, group in sample_risk.groupby('Customer Location'):
        if len(group) > 1:
            tx_list = [f"TX_{str(t)[:8]}" for t in group['Transaction ID']]
            for i in range(len(tx_list)):
                for j in range(i + 1, len(tx_list)):
                    if G.has_edge(tx_list[i], tx_list[j]):
                        G[tx_list[i]][tx_list[j]]['weight'] += 1.0
                        G[tx_list[i]][tx_list[j]]['reasons'].append('shared_location')
                    else:
                        G.add_edge(tx_list[i], tx_list[j], weight=1.0, reasons=['shared_location'])

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_cc = max(components, key=len)
    largest_cc_size = len(largest_cc)

    print(f"    - Transaction Risk Graph: {n_nodes:,} nodes, {n_edges:,} edges")
    print(f"    - Network Density: {density:.6f} | Connected Components: {num_components:,}")
    print(f"    - Largest Connected Component: {largest_cc_size:,} nodes ({largest_cc_size/n_nodes*100:.2f}%)")

    # Step 5: Centrality Analysis
    print("[*] Step 5: Computing Centrality Metrics (Degree, PageRank, Betweenness)...")
    t0 = time.time()
    degree_cent = nx.degree_centrality(G)
    pagerank_vals = nx.pagerank(G, alpha=0.85, weight='weight')
    
    # Compute betweenness on largest connected component or sample for speed
    if largest_cc_size > 500:
        subgraph_lcc = G.subgraph(largest_cc)
        betweenness_vals = nx.betweenness_centrality(subgraph_lcc, k=min(200, largest_cc_size), weight='weight', seed=42)
    else:
        betweenness_vals = nx.betweenness_centrality(G, weight='weight')
    cent_time = time.time() - t0
    print(f"    - Centrality computation completed in {cent_time:.2f}s")

    # Assign centrality attributes to nodes
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_cent.get(node, 0.0)
        G.nodes[node]['pagerank'] = pagerank_vals.get(node, 0.0)
        G.nodes[node]['betweenness'] = betweenness_vals.get(node, 0.0)

    # Step 6: Community Detection & Fraud Ring Discovery
    print("[*] Step 6: Performing Community Detection (Greedy Modularity)...")
    t0 = time.time()
    # Filter to nodes with degree > 0 for modularity
    G_active = G.subgraph([n for n, d in G.degree() if d > 0]).copy()
    if G_active.number_of_edges() > 0:
        communities = list(greedy_modularity_communities(G_active, weight='weight'))
        mod_score = float(modularity(G_active, communities, weight='weight'))
    else:
        communities = [set(G.nodes())]
        mod_score = 0.0
    comm_time = time.time() - t0
    print(f"    - Detected {len(communities)} communities in {comm_time:.2f}s | Modularity Q = {mod_score:.4f}")

    # Analyze Community Risk Profiles
    community_stats = []
    for idx, comm in enumerate(communities):
        comm_nodes = list(comm)
        comm_frauds = sum(G.nodes[n]['is_fraud'] for n in comm_nodes)
        fraud_pct = (comm_frauds / len(comm_nodes)) * 100.0
        avg_amt = np.mean([G.nodes[n]['amount'] for n in comm_nodes])
        avg_pr = np.mean([G.nodes[n]['pagerank'] for n in comm_nodes])
        community_stats.append({
            'community_id': idx + 1,
            'size': len(comm_nodes),
            'fraud_count': comm_frauds,
            'legit_count': len(comm_nodes) - comm_frauds,
            'fraud_rate_pct': round(fraud_pct, 2),
            'avg_amount': round(float(avg_amt), 2),
            'avg_pagerank': round(float(avg_pr), 6)
        })

    # Sort communities by fraud rate and size
    comm_df = pd.DataFrame(community_stats).sort_values(by=['fraud_rate_pct', 'size'], ascending=False)
    print("\n    Top 5 High-Risk Fraud Communities (Fraud Rings):")
    for _, cr in comm_df.head(5).iterrows():
        print(f"    - Community {int(cr['community_id'])}: Size={int(cr['size'])}, Frauds={int(cr['fraud_count'])} ({cr['fraud_rate_pct']}%), Avg Amount=${cr['avg_amount']}")

    # Step 7: Compare Centrality between Fraud and Legitimate Nodes
    fraud_pr = [G.nodes[n]['pagerank'] for n in G.nodes() if G.nodes[n]['is_fraud'] == 1]
    legit_pr = [G.nodes[n]['pagerank'] for n in G.nodes() if G.nodes[n]['is_fraud'] == 0]
    fraud_deg = [G.nodes[n]['degree_centrality'] for n in G.nodes() if G.nodes[n]['is_fraud'] == 1]
    legit_deg = [G.nodes[n]['degree_centrality'] for n in G.nodes() if G.nodes[n]['is_fraud'] == 0]

    mean_pr_fraud = np.mean(fraud_pr) if fraud_pr else 0
    mean_pr_legit = np.mean(legit_pr) if legit_pr else 0
    mean_deg_fraud = np.mean(fraud_deg) if fraud_deg else 0
    mean_deg_legit = np.mean(legit_deg) if legit_deg else 0

    print(f"\n    Centrality Disparity:")
    print(f"    - PageRank       -> Fraud: {mean_pr_fraud:.6f} vs Legit: {mean_pr_legit:.6f} (Disparity: {mean_pr_fraud/max(mean_pr_legit, 1e-9):.2f}x)")
    print(f"    - Degree Cent    -> Fraud: {mean_deg_fraud:.6f} vs Legit: {mean_deg_legit:.6f} (Disparity: {mean_deg_fraud/max(mean_deg_legit, 1e-9):.2f}x)")

    # Step 8: Generate Publication Diagrams
    print("\n[*] Step 7: Generating Publication Diagrams in diagrams/...")

    # Diagram 1: Network Topology
    fig, ax = plt.subplots(figsize=(14, 10))
    # Layout using Spring layout on active component
    layout_graph = G.subgraph(largest_cc) if largest_cc_size <= 400 else G.subgraph(list(largest_cc)[:400])
    pos = nx.spring_layout(layout_graph, k=0.18, iterations=50, seed=42)

    node_colors = []
    node_sizes = []
    for n in layout_graph.nodes():
        if layout_graph.nodes[n]['is_fraud'] == 1:
            node_colors.append('#e41a1c') # Red for fraud
            node_sizes.append(layout_graph.nodes[n]['pagerank'] * 120000 + 40)
        else:
            node_colors.append('#377eb8') # Blue for legitimate
            node_sizes.append(layout_graph.nodes[n]['pagerank'] * 120000 + 20)

    # Draw edges
    nx.draw_networkx_edges(layout_graph, pos, alpha=0.25, edge_color='#999999', width=0.8, ax=ax)
    # Draw nodes
    nx.draw_networkx_nodes(layout_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, ax=ax)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Fraudulent Transaction (Label 1)', markerfacecolor='#e41a1c', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Legitimate Transaction (Label 0)', markerfacecolor='#377eb8', markersize=8),
        Line2D([0], [0], color='#999999', lw=1.5, label='Shared Infrastructure Edge (IP Subnet / Location)'),
        Line2D([0], [0], marker='o', color='w', label='Node Size = PageRank Centrality', markerfacecolor='gray', markersize=12)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    ax.set_title('Transaction Risk Network Topology: Fraud Rings & Infrastructure Sharing Hubs', fontweight='bold', fontsize=13)
    ax.axis('off')

    plt.tight_layout()
    topo_path = 'diagrams/graph_network_topology.png'
    plt.savefig(topo_path, dpi=300)
    plt.close()
    print(f"    - Saved: {topo_path}")

    # Diagram 2: Centrality Analysis & Community Profiling
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Subplot 1: PageRank Distribution
    sns.kdeplot(data=legit_pr, label='Legitimate', color='#377eb8', fill=True, alpha=0.3, ax=axes[0, 0])
    sns.kdeplot(data=fraud_pr, label='Fraudulent', color='#e41a1c', fill=True, alpha=0.3, ax=axes[0, 0])
    axes[0, 0].set_title('PageRank Centrality Distribution (KDE)', fontweight='bold')
    axes[0, 0].set_xlabel('PageRank Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Degree Centrality Boxplot
    cent_df = pd.DataFrame([
        {'Centrality': G.nodes[n]['degree_centrality'], 'Type': 'Fraud' if G.nodes[n]['is_fraud'] == 1 else 'Legitimate'}
        for n in G.nodes()
    ])
    sns.boxplot(x='Type', y='Centrality', data=cent_df, palette={'Fraud': '#e41a1c', 'Legitimate': '#377eb8'}, ax=axes[0, 1])
    axes[0, 1].set_title('Degree Centrality: Fraud vs Legitimate Transactions', fontweight='bold')
    axes[0, 1].set_ylabel('Degree Centrality')
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Top 10 Communities by Fraud Rate
    top10_comm = comm_df.head(10)
    bars = axes[1, 0].bar(range(len(top10_comm)), top10_comm['fraud_rate_pct'], color='#d95f02')
    axes[1, 0].axhline(y=total_fraud/total_tx*100, color='black', linestyle='--', label=f'Baseline Rate ({total_fraud/total_tx*100:.1f}%)')
    axes[1, 0].set_xticks(range(len(top10_comm)))
    axes[1, 0].set_xticklabels([f"C{int(cid)}" for cid in top10_comm['community_id']], rotation=45)
    axes[1, 0].set_title('Fraud Concentration across Detected Communities (Fraud Rings)', fontweight='bold')
    axes[1, 0].set_xlabel('Community ID')
    axes[1, 0].set_ylabel('Fraud Rate (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    # Subplot 4: Centrality vs Transaction Amount Scatter
    scatter_df = pd.DataFrame([
        {
            'Amount': G.nodes[n]['amount'],
            'PageRank': G.nodes[n]['pagerank'],
            'Status': 'Fraud' if G.nodes[n]['is_fraud'] == 1 else 'Legitimate'
        }
        for n in G.nodes()
    ])
    sns.scatterplot(
        x='Amount', y='PageRank', hue='Status', style='Status',
        palette={'Fraud': '#e41a1c', 'Legitimate': '#377eb8'},
        alpha=0.7, data=scatter_df, ax=axes[1, 1]
    )
    axes[1, 1].set_title('Transaction Amount vs PageRank by Fraud Status', fontweight='bold')
    axes[1, 1].set_xlabel('Transaction Amount ($)')
    axes[1, 1].set_ylabel('PageRank Centrality')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Graph Analytics: Centrality Profiles & Fraud Ring Community Risk', fontweight='bold', y=0.98)
    plt.tight_layout()
    cent_path = 'diagrams/graph_centrality_analysis.png'
    plt.savefig(cent_path, dpi=300)
    plt.close()
    print(f"    - Saved: {cent_path}")

    # Step 9: Save Results to JSON & TXT
    print("\n[*] Step 8: Saving Graph Analytics Results to JSON & TXT...")
    graph_output = {
        'metadata': {
            'student_name': 'Kevin Sebastian',
            'nim': '2702746012',
            'course': 'COMP8035041 - Big Data Analytics',
            'total_transactions_analyzed': total_tx,
            'fraud_count': total_fraud,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'heterogeneous_graph': {
            'total_nodes': total_h_nodes,
            'total_edges': total_h_edges
        },
        'transaction_risk_network': {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': float(density),
            'connected_components': num_components,
            'largest_connected_component_size': largest_cc_size,
            'modularity_q': float(mod_score),
            'total_communities': len(communities)
        },
        'centrality_metrics': {
            'mean_pagerank_fraud': float(mean_pr_fraud),
            'mean_pagerank_legit': float(mean_pr_legit),
            'pagerank_disparity_ratio': float(mean_pr_fraud / max(mean_pr_legit, 1e-9)),
            'mean_degree_fraud': float(mean_deg_fraud),
            'mean_degree_legit': float(mean_deg_legit),
            'degree_disparity_ratio': float(mean_deg_fraud / max(mean_deg_legit, 1e-9))
        },
        'top_high_risk_communities': community_stats[:10]
    }

    with open('graph_analytics_results.json', 'w') as f:
        json.dump(graph_output, f, indent=2)
    print("    - Saved: graph_analytics_results.json")

    # TXT Report
    txt_lines = []
    txt_lines.append("=" * 95)
    txt_lines.append(f"{'GRAPH ANALYTICS & FRAUD RING DISCOVERY RESULTS - ASSIGNMENT II':^95}")
    txt_lines.append(f"{'Student: Kevin Sebastian | NIM: 2702746012':^95}")
    txt_lines.append("=" * 95)
    txt_lines.append(f"1. HETEROGENEOUS ENTITY GRAPH:")
    txt_lines.append(f"   * Total Nodes (Transactions, Subnets, Locations, States) : {total_h_nodes:,}")
    txt_lines.append(f"   * Total Edges (Entity Sharing & Ingestion Links)         : {total_h_edges:,}")
    txt_lines.append(f"\n2. PROJECTED TRANSACTION RISK NETWORK:")
    txt_lines.append(f"   * Total Nodes (High-Risk & Fraud Connected)              : {n_nodes:,}")
    txt_lines.append(f"   * Total Edges (Shared Subnet / Location Associations)    : {n_edges:,}")
    txt_lines.append(f"   * Network Density                                        : {density:.6f}")
    txt_lines.append(f"   * Connected Components                                   : {num_components:,}")
    txt_lines.append(f"   * Largest Connected Component Size                       : {largest_cc_size:,} nodes ({largest_cc_size/n_nodes*100:.2f}%)")
    txt_lines.append(f"   * Modularity Score (Q)                                   : {mod_score:.4f}")
    txt_lines.append(f"   * Detected Communities / Fraud Clusters                  : {len(communities)}")
    txt_lines.append(f"\n3. CENTRALITY DISPARITY (FRAUD vs LEGITIMATE):")
    txt_lines.append(f"   * Mean PageRank Centrality   -> Fraud: {mean_pr_fraud:.6f} | Legit: {mean_pr_legit:.6f} | Ratio: {mean_pr_fraud/max(mean_pr_legit, 1e-9):.2f}x")
    txt_lines.append(f"   * Mean Degree Centrality     -> Fraud: {mean_deg_fraud:.6f} | Legit: {mean_deg_legit:.6f} | Ratio: {mean_deg_fraud/max(mean_deg_legit, 1e-9):.2f}x")
    txt_lines.append(f"\n4. TOP DETECTED HIGH-RISK COMMUNITIES (FRAUD RINGS):")
    txt_lines.append(f"   {'Comm ID':<8} | {'Size':<6} | {'Frauds':<8} | {'Legit':<6} | {'Fraud Rate (%)':<15} | {'Avg Amount ($)':<15} | {'Avg PageRank':<12}")
    txt_lines.append(f"   {'-'*75}")
    for cr in community_stats[:10]:
        txt_lines.append(
            f"   {cr['community_id']:<8} | {cr['size']:<6} | {cr['fraud_count']:<8} | {cr['legit_count']:<6} | "
            f"{cr['fraud_rate_pct']:<15.2f} | ${cr['avg_amount']:<14.2f} | {cr['avg_pagerank']:<12.6f}"
        )
    txt_lines.append("=" * 95)

    txt_content = "\n".join(txt_lines)
    with open('graph_analytics_results.txt', 'w') as f:
        f.write(txt_content)
    print("    - Saved: graph_analytics_results.txt")
    print("\n" + txt_content)
    print("\n[SUCCESS] Task 3: Graph Analytics execution finished successfully!")

if __name__ == '__main__':
    run_graph_analytics()
