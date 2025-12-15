"""
Building Importance Analysis using PageRank
Computes importance scores for each building in the Graph City
"""

import pandas as pd
import networkx as nx
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent.parent
GRAPH_DATA_DIR = BASE_DIR / "graph_data"
ANALYSIS_DIR = BASE_DIR / "analysis"
VIZ_DIR = BASE_DIR / "visualizations"

# Create directories
ANALYSIS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

def load_data():
    """Load graph and user data"""
    print("Loading data...")
    
    edges = pd.read_csv(
        GRAPH_DATA_DIR / "stackoverflow_electronics_main.txt",
        sep='\t',
        header=None,
        names=['source', 'target']
    )
    
    labels = pd.read_csv(
        GRAPH_DATA_DIR / "stackoverflow_electronics_main_labels.csv"
    )
    
    # Load k-core assignments
    user_df = pd.read_csv(ANALYSIS_DIR / "user_k_cores.csv")
    
    return edges, labels, user_df

def assign_users_to_buildings(user_df):
    """
    Assign each user to a building based on their k-core value
    Buildings from your Graph City: 22 buildings with k-cores from 1 to 76
    """
    def get_building(k_core):
        # Map k-core to actual building names from your city
        if k_core >= 76:
            return "wavemap_76_87_1"
        elif k_core >= 51:
            return "wavemap_51_18_1"
        elif k_core >= 32:
            return "wavemap_32_3_1"
        elif k_core >= 18:
            return "wavemap_18_11_1"
        elif k_core >= 13:
            return "wavemap_13_5607_1"
        elif k_core >= 11:
            return "wavemap_11_23_1"
        elif k_core >= 9:
            return "wavemap_9_3097_1"
        elif k_core >= 8:
            return "wavemap_8_3_1"
        elif k_core >= 7:
            return "wavemap_7_13363_2"
        elif k_core >= 6:
            return "wavemap_6_169_1"
        elif k_core >= 5:
            return "wavemap_5_12699_1 or wavemap_5_38_1"
        elif k_core >= 4:
            return "wavemap_4_179_1"
        elif k_core >= 3:
            return "wavemap_3_17881_1 or wavemap_3_1320_1"
        elif k_core >= 2:
            return "wavemap_2_16614_1 or wavemap_2_664_2 or wavemap_2_552_1"
        else:
            return "wavemap_1_17795_20 or wavemap_1_4738_30 or wavemap_1_7607_4 or wavemap_1_568_4"
    
    user_df['building'] = user_df['k_core'].apply(get_building)
    return user_df

def build_meta_graph(edges, user_df):
    """
    Build meta-graph where nodes are buildings
    and edges represent interactions between buildings
    """
    print("\nBuilding meta-graph...")
    
    G_meta = nx.DiGraph()
    
    # Count interactions between buildings
    building_interactions = {}
    
    for _, edge in edges.iterrows():
        source_user = edge['source']
        target_user = edge['target']
        
        # Get buildings for both users
        source_building = user_df[user_df['user_id'] == source_user]['building'].values
        target_building = user_df[user_df['user_id'] == target_user]['building'].values
        
        if len(source_building) > 0 and len(target_building) > 0:
            source_building = source_building[0]
            target_building = target_building[0]
            
            # Handle multiple building options (e.g., "wavemap_1_17795_20 or wavemap_1_568_4")
            if " or " in source_building:
                source_building = source_building.split(" or ")[0]
            if " or " in target_building:
                target_building = target_building.split(" or ")[0]
            
            edge_key = (source_building, target_building)
            building_interactions[edge_key] = building_interactions.get(edge_key, 0) + 1
    
    # Add edges to meta-graph
    for (source, target), weight in building_interactions.items():
        if source != target:  # No self-loops
            G_meta.add_edge(source, target, weight=weight)
    
    print(f"Meta-graph: {G_meta.number_of_nodes()} buildings, {G_meta.number_of_edges()} connections")
    
    return G_meta

def compute_importance(G_meta):
    """Compute importance metrics for each building"""
    print("\nComputing importance metrics...")
    
    # PageRank (most important)
    pagerank = nx.pagerank(G_meta, weight='weight')
    
    # Betweenness centrality (bridge between communities)
    betweenness = nx.betweenness_centrality(G_meta, weight='weight')
    
    # Degree centrality (most connected)
    in_degree = dict(G_meta.in_degree(weight='weight'))
    out_degree = dict(G_meta.out_degree(weight='weight'))
    
    # Combine into dataframe
    buildings = list(pagerank.keys())
    
    importance_df = pd.DataFrame({
        'building': buildings,
        'pagerank': [pagerank[b] for b in buildings],
        'betweenness': [betweenness[b] for b in buildings],
        'in_degree': [in_degree[b] for b in buildings],
        'out_degree': [out_degree[b] for b in buildings],
        'total_degree': [in_degree[b] + out_degree[b] for b in buildings]
    })
    
    # Sort by PageRank
    importance_df = importance_df.sort_values('pagerank', ascending=False)
    
    # Add rank
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return importance_df

def save_results(importance_df, G_meta):
    """Save importance analysis results"""
    # Save to CSV
    importance_df.to_csv(ANALYSIS_DIR / "building_importance.csv", index=False)
    print(f"\nSaved importance scores to {ANALYSIS_DIR / 'building_importance.csv'}")
    
    # Save meta-graph
    nx.write_gexf(G_meta, ANALYSIS_DIR / "building_meta_graph.gexf")
    print(f"Saved meta-graph to {ANALYSIS_DIR / 'building_meta_graph.gexf'}")
    
    # Create summary
    summary = {
        "total_buildings": len(importance_df),
        "total_connections": G_meta.number_of_edges(),
        "most_important": importance_df.iloc[0]['building'],
        "highest_pagerank": float(importance_df.iloc[0]['pagerank']),
        "top_5_buildings": importance_df.head(5)[['building', 'pagerank']].to_dict('records')
    }
    
    with open(ANALYSIS_DIR / "building_importance_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def visualize_meta_graph(G_meta, importance_df):
    """Create visualization of building meta-graph"""
    print("\nCreating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Graph 1: Network visualization
    pos = nx.spring_layout(G_meta, k=3, iterations=50, seed=42)
    
    # Node sizes based on PageRank
    node_sizes = [importance_df[importance_df['building'] == node]['pagerank'].values[0] * 50000 
                  for node in G_meta.nodes()]
    
    # Draw network
    nx.draw_networkx_nodes(G_meta, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7, ax=ax1)
    nx.draw_networkx_edges(G_meta, pos, alpha=0.2, arrows=True, 
                          edge_color='gray', ax=ax1)
    nx.draw_networkx_labels(G_meta, pos, font_size=7, ax=ax1)
    
    ax1.set_title("Building Meta-Graph\n(Node size = PageRank importance)", fontsize=14)
    ax1.axis('off')
    
    # Graph 2: Importance ranking
    top_10 = importance_df.head(10)
    ax2.barh(range(len(top_10)), top_10['pagerank'])
    ax2.set_yticks(range(len(top_10)))
    ax2.set_yticklabels(top_10['building'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('PageRank Score', fontsize=12)
    ax2.set_title('Top 10 Most Important Buildings', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "building_importance.png", dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {VIZ_DIR / 'building_importance.png'}")
    
    plt.close()

def display_results(importance_df):
    """Display key findings"""
    print("\n" + "="*80)
    print("BUILDING IMPORTANCE ANALYSIS (PageRank)")
    print("="*80)
    
    print("\n🏆 TOP 10 MOST IMPORTANT BUILDINGS:")
    print(importance_df[['rank', 'building', 'pagerank', 'betweenness', 'total_degree']].head(10).to_string(index=False))
    
    print("\n🌉 TOP 5 BRIDGE BUILDINGS (Betweenness):")
    print(importance_df.nlargest(5, 'betweenness')[['building', 'betweenness', 'pagerank']].to_string(index=False))
    
    print("\n🔗 TOP 5 MOST CONNECTED BUILDINGS:")
    print(importance_df.nlargest(5, 'total_degree')[['building', 'total_degree', 'pagerank']].to_string(index=False))
    
    print("\n" + "="*80)

def main():
    """Main execution"""
    print("="*80)
    print("BUILDING IMPORTANCE ANALYSIS - PAGERANK")
    print("="*80)
    
    # Load data
    edges, labels, user_df = load_data()
    
    # Assign users to buildings
    user_df = assign_users_to_buildings(user_df)
    
    # Build meta-graph
    G_meta = build_meta_graph(edges, user_df)
    
    # Compute importance
    importance_df = compute_importance(G_meta)
    
    # Save results
    save_results(importance_df, G_meta)
    
    # Visualize
    visualize_meta_graph(G_meta, importance_df)
    
    # Display results
    display_results(importance_df)
    
    print("\n✅ Analysis complete!")
    print(f"Results saved to: {ANALYSIS_DIR}")
    print(f"Visualizations saved to: {VIZ_DIR}")

if __name__ == '__main__':
    main()