"""
AnalyzingGraph.py
Analyze the user interaction graph properties
Check if it's suitable for Graph-Cities visualization
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

print("\n" + "="*70)
print("Analyzing User Interaction Graph")
print("="*70 + "\n")

# Step 1: Load edge list
print("Step 1: Loading graph data...")
edges = pd.read_csv('graph_data/stackoverflow_electronics.txt', 
                    sep='\t', header=None, names=['user1', 'user2'])

print(f"✓ Loaded {len(edges):,} edges")

# Step 2: Build NetworkX graph
print("\nStep 2: Building NetworkX graph...")
G = nx.Graph()
G.add_edges_from(edges.values)

print(f"✓ Graph constructed")

# Step 2.5: Remove self-loops (if any)
print("\nStep 2.5: Checking for self-loops...")
self_loops = list(nx.selfloop_edges(G))
if len(self_loops) > 0:
    print(f"⚠️  Found {len(self_loops)} self-loops, removing them...")
    G.remove_edges_from(self_loops)
    print(f"✓ Removed self-loops")
else:
    print(f"✓ No self-loops found")

# Step 3: Basic statistics
print("\n" + "="*70)
print("BASIC GRAPH STATISTICS")
print("="*70)

print(f"\nNodes (Users): {G.number_of_nodes():,}")
print(f"Edges (Connections): {G.number_of_edges():,}")
print(f"Density: {nx.density(G):.6f}")

# Degree statistics
degrees = dict(G.degree())
degree_values = list(degrees.values())

print(f"\nDegree Statistics:")
print(f"  Average degree: {sum(degree_values) / len(degree_values):.2f}")
print(f"  Median degree: {sorted(degree_values)[len(degree_values)//2]}")
print(f"  Max degree: {max(degree_values)}")
print(f"  Min degree: {min(degree_values)}")

# Step 4: Connected components analysis
print("\n" + "="*70)
print("CONNECTED COMPONENTS ANALYSIS")
print("="*70)

components = list(nx.connected_components(G))
print(f"\nTotal connected components: {len(components)}")

# Sort by size
component_sizes = sorted([len(c) for c in components], reverse=True)
print(f"\nTop 10 component sizes:")
for i, size in enumerate(component_sizes[:10], 1):
    print(f"  {i}. {size:,} nodes ({size/G.number_of_nodes()*100:.2f}%)")

# Largest component
largest_cc = max(components, key=len)
print(f"\n✓ Largest component: {len(largest_cc):,} nodes ({len(largest_cc)/G.number_of_nodes()*100:.2f}% of total)")

# Step 5: Extract largest component
print("\nStep 3: Extracting largest connected component...")
G_main = G.subgraph(largest_cc).copy()

print(f"Main component statistics:")
print(f"  Nodes: {G_main.number_of_nodes():,}")
print(f"  Edges: {G_main.number_of_edges():,}")
print(f"  Density: {nx.density(G_main):.6f}")

# Step 6: Clustering coefficient
print("\n" + "="*70)
print("CLUSTERING ANALYSIS")
print("="*70)

print("\nCalculating clustering coefficient (this may take a moment)...")
avg_clustering = nx.average_clustering(G_main)
print(f"✓ Average clustering coefficient: {avg_clustering:.4f}")
print(f"  (Higher values indicate more tightly-knit communities)")

# Step 7: Degree distribution
print("\n" + "="*70)
print("DEGREE DISTRIBUTION ANALYSIS")
print("="*70)

main_degrees = [d for n, d in G_main.degree()]
print(f"\nMain component degree statistics:")
print(f"  Average: {sum(main_degrees) / len(main_degrees):.2f}")
print(f"  Median: {sorted(main_degrees)[len(main_degrees)//2]}")
print(f"  Max: {max(main_degrees)}")

# Check for power-law behavior
degree_freq = pd.Series(main_degrees).value_counts().sort_index()
print(f"\n✓ Degree distribution calculated")

# Step 8: Core number (k-core) analysis
print("\n" + "="*70)
print("K-CORE ANALYSIS (Important for Graph-Cities!)")
print("="*70)

print("\nCalculating k-cores...")
core_numbers = nx.core_number(G_main)
max_core = max(core_numbers.values())

print(f"\n✓ Maximum k-core number: {max_core}")
print(f"  (This determines the 'height' of your Graph City)")

# Count nodes at each core level
core_distribution = pd.Series(core_numbers.values()).value_counts().sort_index()
print(f"\nK-core distribution (top 15 levels):")
top_cores = sorted(core_distribution.index, reverse=True)[:15]
for k in top_cores:
    count = core_distribution[k]
    print(f"  k={k:3d}: {count:6,} nodes ({count/len(core_numbers)*100:.2f}%)")

# Step 9: Save filtered graph (main component only, no self-loops)
print("\n" + "="*70)
print("SAVING PROCESSED GRAPH")
print("="*70)

output_file = 'graph_data/stackoverflow_electronics_main.txt'
print(f"\nSaving main component to: {output_file}")

with open(output_file, 'w') as f:
    for u, v in G_main.edges():
        f.write(f"{u}\t{v}\n")

print(f"✓ Saved {G_main.number_of_edges():,} edges")

# Save labels for main component only
print("\nFiltering user labels to main component...")
main_nodes = set(G_main.nodes())
labels_df = pd.read_csv('graph_data/stackoverflow_electronics_labels.csv', 
                        header=None, names=['UserId', 'DisplayName'])
labels_main = labels_df[labels_df['UserId'].isin(main_nodes)]
labels_main.to_csv('graph_data/stackoverflow_electronics_main_labels.csv', 
                   index=False, header=False)
print(f"✓ Saved {len(labels_main):,} labels")

# Step 10: Create visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Degree distribution
axes[0, 0].hist(main_degrees, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_xlabel('Degree')
axes[0, 0].set_ylabel('Number of Nodes')
axes[0, 0].set_title('Degree Distribution')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# 2. Degree distribution (log-log for power law)
axes[0, 1].loglog(degree_freq.index, degree_freq.values, 'o-', alpha=0.7, color='darkred')
axes[0, 1].set_xlabel('Degree (k)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Degree Distribution (Log-Log)')
axes[0, 1].grid(True, alpha=0.3)

# 3. K-core distribution
axes[1, 0].bar(core_distribution.index, core_distribution.values, 
               edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_xlabel('Core Number (k)')
axes[1, 0].set_ylabel('Number of Nodes')
axes[1, 0].set_title('K-Core Distribution')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# 4. Component size distribution
axes[1, 1].bar(range(1, min(21, len(component_sizes)+1)), 
               component_sizes[:20], 
               edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].set_xlabel('Component Rank')
axes[1, 1].set_ylabel('Component Size (nodes)')
axes[1, 1].set_title('Connected Component Sizes (Top 20)')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/12_graph_structure_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: visualizations/12_graph_structure_analysis.png")
plt.close()

# Step 11: Save analysis summary
print("\nSaving analysis summary...")

analysis_summary = {
    'total_nodes': G.number_of_nodes(),
    'total_edges': G.number_of_edges(),
    'density': float(nx.density(G)),
    'num_components': len(components),
    'largest_component_nodes': len(largest_cc),
    'largest_component_edges': G_main.number_of_edges(),
    'largest_component_percentage': len(largest_cc) / G.number_of_nodes() * 100,
    'avg_degree': sum(main_degrees) / len(main_degrees),
    'max_degree': max(main_degrees),
    'avg_clustering': float(avg_clustering),
    'max_k_core': int(max_core),
    'suitable_for_graph_cities': len(largest_cc) > 100 and max_core > 3
}

with open('graph_data/graph_analysis.json', 'w') as f:
    json.dump(analysis_summary, f, indent=2)

print(f"✓ Saved: graph_data/graph_analysis.json")

# Step 12: Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS FOR GRAPH-CITIES")
print("="*70)

print(f"\n✓ Graph quality assessment:")

if len(largest_cc) / G.number_of_nodes() > 0.8:
    print(f"  ✅ EXCELLENT: Large main component ({len(largest_cc)/G.number_of_nodes()*100:.1f}% of nodes)")
else:
    print(f"  ⚠️  WARNING: Main component is only {len(largest_cc)/G.number_of_nodes()*100:.1f}% of nodes")
    print(f"     Consider using only the main component")

if max_core >= 10:
    print(f"  ✅ EXCELLENT: Very high k-core ({max_core}) - will create very tall buildings!")
elif max_core >= 5:
    print(f"  ✅ GOOD: High k-core ({max_core}) - will create tall buildings")
elif max_core >= 3:
    print(f"  ✓ ACCEPTABLE: Moderate k-core ({max_core}) - will create medium buildings")
else:
    print(f"  ⚠️  WARNING: Low k-core ({max_core}) - buildings may be short")

if avg_clustering > 0.5:
    print(f"  ✅ EXCELLENT: Very high clustering ({avg_clustering:.3f}) - extremely strong communities!")
elif avg_clustering > 0.3:
    print(f"  ✅ GOOD: High clustering ({avg_clustering:.3f}) - strong communities")
elif avg_clustering > 0.1:
    print(f"  ✓ ACCEPTABLE: Moderate clustering ({avg_clustering:.3f}) - visible communities")
else:
    print(f"  ⚠️  INFO: Low clustering ({avg_clustering:.3f}) - weaker community structure")

print(f"\n📊 Recommended file for Graph-Cities:")
if len(largest_cc) / G.number_of_nodes() > 0.9:
    print(f"   Use: stackoverflow_electronics.txt (full graph)")
    print(f"   But also generated: stackoverflow_electronics_main.txt (cleaned)")
else:
    print(f"   Use: stackoverflow_electronics_main.txt (main component only)")

print(f"\n🎯 Expected Graph City characteristics:")
print(f"   • City will have ~{len(largest_cc):,} buildings")
print(f"   • Tallest buildings will be k={max_core} stories high")
print(f"   • Strong community clustering (coefficient: {avg_clustering:.3f})")
print(f"   • Hub-and-spoke structure (max degree: {max(main_degrees)})")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
print(f"\nFiles created:")
print(f"  1. graph_data/stackoverflow_electronics_main.txt (CLEANED - use this!)")
print(f"  2. graph_data/stackoverflow_electronics_main_labels.csv")
print(f"  3. graph_data/graph_analysis.json")
print(f"  4. visualizations/12_graph_structure_analysis.png")

print(f"\n🚀 Next step: Set up Graph-Cities repository and run the pipeline!")
print(f"   Your graph is {'EXCELLENT' if max_core >= 10 and avg_clustering > 0.5 else 'GOOD'} for Graph-Cities visualization!")
print("="*70 + "\n")