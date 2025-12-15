"""
BuildingGraph.py
Build user-user interaction graph from Stack Overflow data
Users connect if they both answered the same question
"""

from LoadingRawData import load_posts, load_users
import pandas as pd
from itertools import combinations
from collections import Counter
import os

print("\n" + "="*70)
print("Building User Interaction Graph")
print("="*70 + "\n")

# Create output directory
os.makedirs('graph_data', exist_ok=True)

# Step 1: Load data
print("Step 1: Loading data...")
posts = load_posts()
users = load_users()

# Step 2: Convert IDs to numeric for matching
print("\nStep 2: Preprocessing data...")
posts['Id'] = pd.to_numeric(posts['Id'], errors='coerce')
posts['PostTypeId'] = pd.to_numeric(posts['PostTypeId'], errors='coerce')
posts['ParentId'] = pd.to_numeric(posts['ParentId'], errors='coerce')
posts['OwnerUserId'] = pd.to_numeric(posts['OwnerUserId'], errors='coerce')

# Step 3: Filter to questions and answers
questions = posts[posts['PostTypeId'] == 1].copy()
answers = posts[posts['PostTypeId'] == 2].copy()

print(f"  Questions: {len(questions):,}")
print(f"  Answers: {len(answers):,}")

# Step 4: Build edges (users who answered same question)
print("\nStep 3: Building edges (users who co-answered questions)...")

edges = []
questions_with_multi_answers = 0

# Group answers by ParentId (question they're answering)
answer_groups = answers.groupby('ParentId')['OwnerUserId'].apply(list)

for question_id, answerers in answer_groups.items():
    # Remove NaN users
    answerers = [user for user in answerers if pd.notna(user)]
    
    # Only create edges if 2+ users answered
    if len(answerers) >= 2:
        questions_with_multi_answers += 1
        
        # Create edges between all pairs
        for user1, user2 in combinations(answerers, 2):
            # Sort to avoid duplicates (1,2) and (2,1)
            edge = (int(min(user1, user2)), int(max(user1, user2)))
            edges.append(edge)
        
        # Progress indicator
        if questions_with_multi_answers % 1000 == 0:
            print(f"  Processed {questions_with_multi_answers:,} questions with multiple answers...")

print(f"\n✓ Found {questions_with_multi_answers:,} questions with multiple answers")
print(f"✓ Generated {len(edges):,} raw edges")

# Step 5: Count edge weights (how many questions they co-answered)
print("\nStep 4: Counting edge frequencies...")
edge_counts = Counter(edges)
print(f"✓ Unique edges: {len(edge_counts):,}")

# Step 6: Save edge list
print("\nStep 5: Saving edge list...")
edge_file = 'graph_data/stackoverflow_electronics.txt'
with open(edge_file, 'w') as f:
    for (u1, u2), weight in edge_counts.items():
        # For now, write unweighted edges (Graph-Cities expects simple edge list)
        # If you want weighted, repeat the edge 'weight' times or modify format
        f.write(f"{u1}\t{u2}\n")

print(f"✓ Saved to: {edge_file}")

# Step 7: Get statistics about the graph
print("\nStep 6: Graph Statistics...")
unique_users = set()
for (u1, u2) in edge_counts.keys():
    unique_users.add(u1)
    unique_users.add(u2)

print(f"  Total edges: {len(edge_counts):,}")
print(f"  Users in graph: {len(unique_users):,} (out of {len(users):,} total)")
print(f"  Avg edges per user: {(len(edge_counts) * 2) / len(unique_users):.2f}")

# Find most connected users
user_degree = Counter()
for (u1, u2) in edge_counts.keys():
    user_degree[u1] += 1
    user_degree[u2] += 1

top_connected = user_degree.most_common(10)
print(f"\n  Top 10 Most Connected Users:")
for user_id, degree in top_connected:
    # Try to get user name
    user_info = users[pd.to_numeric(users['Id'], errors='coerce') == user_id]
    if not user_info.empty:
        name = user_info.iloc[0]['DisplayName']
        print(f"    User {user_id} ({name}): {degree} connections")
    else:
        print(f"    User {user_id}: {degree} connections")

# Step 8: Save user labels
print("\nStep 7: Saving user labels...")
users['Id_numeric'] = pd.to_numeric(users['Id'], errors='coerce')
user_labels = users[users['Id_numeric'].isin(unique_users)][['Id_numeric', 'DisplayName']]
user_labels = user_labels.dropna()

label_file = 'graph_data/stackoverflow_electronics_labels.csv'
user_labels.to_csv(label_file, index=False, header=False)
print(f"✓ Saved {len(user_labels):,} user labels to: {label_file}")

# Step 9: Save summary statistics
print("\nStep 8: Saving summary statistics...")
summary = {
    'total_users': len(users),
    'users_in_graph': len(unique_users),
    'total_edges': len(edge_counts),
    'questions_with_multi_answers': questions_with_multi_answers,
    'avg_degree': (len(edge_counts) * 2) / len(unique_users)
}

import json
with open('graph_data/graph_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("✅ GRAPH CONSTRUCTION COMPLETE!")
print("="*70)
print(f"\nOutput files created in graph_data/:")
print(f"  1. {edge_file} - Edge list for Graph-Cities")
print(f"  2. {label_file} - User ID to name mapping")
print(f"  3. graph_data/graph_summary.json - Statistics")
print("\nNext step: Run AnalyzingGraph.py to examine graph properties")
print("="*70 + "\n")