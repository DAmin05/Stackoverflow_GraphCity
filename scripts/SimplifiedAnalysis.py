"""
SimplifiedAnalysis.py
Analyze communities without Graph-Cities 3D visualization
Uses NetworkX k-core decomposition instead
"""

import networkx as nx
import pandas as pd
import json
import os
from collections import Counter, defaultdict

print("="*70)
print("SIMPLIFIED COMMUNITY ANALYSIS (Without Graph-Cities)")
print("="*70)

# Get correct path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
graph_file = os.path.join(project_dir, 'graph_data', 'stackoverflow_electronics_main.txt')

print(f"\nLooking for file: {graph_file}")
print(f"File exists: {os.path.exists(graph_file)}")

print("\nStep 1: Loading graph...")
edges = pd.read_csv(graph_file, sep='\t', header=None, names=['user1', 'user2'])

G = nx.Graph()
G.add_edges_from(edges.values)

print(f"✓ Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# K-core decomposition
print("\nStep 2: Computing k-core decomposition...")
core_numbers = nx.core_number(G)
max_k = max(core_numbers.values())
print(f"✓ Maximum k-core: {max_k}")

# Group users by k-core level
cores = defaultdict(list)
for user, k in core_numbers.items():
    cores[k].append(user)

print(f"✓ Found {len(cores)} different k-core levels")

# Get k-core subgraphs (top 15 cores)
print("\nStep 3: Extracting k-core communities (top 15)...")
buildings = {}
for k in sorted(cores.keys(), reverse=True)[:15]:
    k_core = nx.k_core(G, k)
    if k_core.number_of_nodes() > 0:
        buildings[f"core_{k}"] = {
            'k_value': int(k),  # Convert to int
            'num_users': int(k_core.number_of_nodes()),  # Convert to int
            'num_edges': int(k_core.number_of_edges()),  # Convert to int
            'density': float(nx.density(k_core)),  # Convert to float
            'users': [int(u) for u in k_core.nodes()]  # Convert all to int
        }
        print(f"  k={k:3d}: {k_core.number_of_nodes():6,} users, {k_core.number_of_edges():7,} edges, density={nx.density(k_core):.4f}")

# Save results
print("\nStep 4: Saving results...")
analysis_dir = os.path.join(project_dir, 'analysis')
os.makedirs(analysis_dir, exist_ok=True)

output_file = os.path.join(analysis_dir, 'simplified_buildings.json')
with open(output_file, 'w') as f:
    json.dump(buildings, f, indent=2)

print(f"✓ Saved {len(buildings)} communities to {output_file}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total communities analyzed: {len(buildings)}")
print(f"Largest community (k={max_k}): {buildings[f'core_{max_k}']['num_users']:,} users")
print(f"Total users in top 15 cores: {sum(b['num_users'] for b in buildings.values()):,}")
print("="*70)
print("\n✓ Analysis complete!")
print(f"\nNext step: Run TFIDFAnalysis.py to identify tag characteristics for each community")
