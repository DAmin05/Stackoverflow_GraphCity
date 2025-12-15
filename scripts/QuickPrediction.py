"""
Quick K-Core Analysis and User Classification
Analyzes user importance based on k-core decomposition
"""

import pandas as pd
import networkx as nx
import json
from pathlib import Path

# Define paths relative to scripts directory
BASE_DIR = Path(__file__).parent.parent
GRAPH_DATA_DIR = BASE_DIR / "graph_data"
ANALYSIS_DIR = BASE_DIR / "analysis"

# Create analysis directory if it doesn't exist
ANALYSIS_DIR.mkdir(exist_ok=True)

def load_graph():
    """Load graph from edge list and labels"""
    print("Loading graph data...")
    
    # Load edge list
    edges = pd.read_csv(
        GRAPH_DATA_DIR / "stackoverflow_electronics_main.txt",
        sep='\t',
        header=None,
        names=['source', 'target']
    )
    
    # Load labels
    labels = pd.read_csv(
        GRAPH_DATA_DIR / "stackoverflow_electronics_main_labels.csv"
    )
    
    # Check what columns we have
    print(f"Label columns: {list(labels.columns)}")
    
    # Rename columns to standardize
    # Handle different possible column names
    if 'new_id' in labels.columns:
        labels = labels.rename(columns={'new_id': 'user_id'})
    elif labels.columns[0] != 'user_id':
        # First column is user_id
        labels.columns = ['user_id'] + list(labels.columns[1:])
    
    print(f"Loaded {len(edges)} edges and {len(labels)} users")
    
    return edges, labels

def compute_k_cores(edges):
    """Compute k-core values for all users"""
    print("\nBuilding NetworkX graph...")
    G = nx.Graph()
    
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'])
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print("Computing k-core decomposition...")
    k_cores = nx.core_number(G)
    
    return k_cores, G

def analyze_users(k_cores, labels, G):
    """Create comprehensive user analysis"""
    print("\nAnalyzing users...")
    
    # Create user dataframe
    user_df = pd.DataFrame({
        'user_id': list(k_cores.keys()),
        'k_core': list(k_cores.values())
    })
    
    # Add degree (number of connections)
    user_df['degree'] = user_df['user_id'].apply(lambda x: G.degree(x) if x in G else 0)
    
    # Merge with labels (handle different column structures)
    if 'user_id' in labels.columns:
        user_df = user_df.merge(labels, on='user_id', how='left')
    else:
        # If labels doesn't have user_id, assume first column is user_id
        labels_copy = labels.copy()
        labels_copy.insert(0, 'user_id', range(len(labels_copy)))
        user_df = user_df.merge(labels_copy, on='user_id', how='left')
    
    # Classify users by expertise level
    def classify_user(k_core):
        if k_core >= 70:
            return "Elite Expert"
        elif k_core >= 50:
            return "Senior Expert"
        elif k_core >= 30:
            return "Expert"
        elif k_core >= 15:
            return "Advanced"
        elif k_core >= 5:
            return "Intermediate"
        else:
            return "Beginner"
    
    user_df['expertise_level'] = user_df['k_core'].apply(classify_user)
    
    # Sort by k-core
    user_df = user_df.sort_values('k_core', ascending=False)
    
    return user_df

def predict_building(k_core_value):
    """
    Predict which building (community) a user belongs to based on k-core
    
    Buildings are named: wavemap_{k-core}_{community-id}_{wave}
    """
    if k_core_value >= 70:
        return f"wavemap_76_87_1 (Elite Core)"
    elif k_core_value >= 50:
        return f"wavemap_51_18_1 (Senior Experts)"
    elif k_core_value >= 30:
        return f"wavemap_32_3_1 (Expert Community)"
    elif k_core_value >= 15:
        return f"wavemap_18_11_1 (Advanced Users)"
    elif k_core_value >= 10:
        return f"wavemap_13_5607_1 or wavemap_11_23_1 (Active Contributors)"
    elif k_core_value >= 5:
        return f"wavemap_5_38_1 to wavemap_9_3097_1 (Regular Users)"
    elif k_core_value >= 2:
        return f"wavemap_2_552_1 to wavemap_3_1320_1 (Occasional Users)"
    else:
        return f"wavemap_1_17795_20 or wavemap_1_568_4 (Peripheral Users)"

def save_results(user_df):
    """Save analysis results"""
    # Save full user data
    user_df.to_csv(ANALYSIS_DIR / "user_k_cores.csv", index=False)
    print(f"\nSaved user k-cores to {ANALYSIS_DIR / 'user_k_cores.csv'}")
    
    # Create summary statistics
    summary = {
        "total_users": int(len(user_df)),
        "max_k_core": int(user_df['k_core'].max()),
        "min_k_core": int(user_df['k_core'].min()),
        "avg_k_core": float(user_df['k_core'].mean()),
        "median_k_core": float(user_df['k_core'].median()),
        "expertise_distribution": {k: int(v) for k, v in user_df['expertise_level'].value_counts().to_dict().items()}
    }
    
    with open(ANALYSIS_DIR / "k_core_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {ANALYSIS_DIR / 'k_core_summary.json'}")

def display_results(user_df):
    """Display key findings"""
    print("\n" + "="*80)
    print("USER K-CORE ANALYSIS RESULTS")
    print("="*80)
    
    print("\n📊 K-CORE DISTRIBUTION:")
    k_core_dist = user_df['k_core'].value_counts().sort_index(ascending=False).head(15)
    for k, count in k_core_dist.items():
        print(f"  k={k}: {count} users")
    
    print("\n👥 EXPERTISE LEVEL DISTRIBUTION:")
    for level, count in user_df['expertise_level'].value_counts().items():
        print(f"  {level}: {count} users")
    
    print("\n🏆 TOP 20 USERS (Elite Experts):")
    display_cols = ['user_id', 'k_core', 'degree', 'expertise_level']
    # Add original_user_id if it exists
    if 'original_user_id' in user_df.columns:
        display_cols.insert(1, 'original_user_id')
    
    print(user_df[display_cols].head(20).to_string(index=False))
    
    print("\n🏙️ BUILDING PREDICTIONS:")
    print("\nTop users would belong to these buildings:")
    for _, user in user_df.head(10).iterrows():
        building = predict_building(user['k_core'])
        user_id_str = f"User {user['user_id']}"
        if 'original_user_id' in user_df.columns:
            user_id_str += f" (Original ID: {user.get('original_user_id', 'N/A')})"
        print(f"  {user_id_str} (k-core={user['k_core']}): {building}")
    
    print("\n" + "="*80)

def main():
    """Main execution"""
    print("="*80)
    print("STACK OVERFLOW ELECTRONICS - USER K-CORE ANALYSIS")
    print("="*80)
    
    # Load data
    edges, labels = load_graph()
    
    # Compute k-cores
    k_cores, G = compute_k_cores(edges)
    
    # Analyze users
    user_df = analyze_users(k_cores, labels, G)
    
    # Save results
    save_results(user_df)
    
    # Display results
    display_results(user_df)
    
    print("\n✅ Analysis complete!")
    print(f"Results saved to: {ANALYSIS_DIR}")

if __name__ == '__main__':
    main()