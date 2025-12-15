"""
VisualizingData.py
Create comprehensive visualizations and statistical analysis for Stack Overflow data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
from LoadingRawData import load_users, load_posts, load_comments
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Create output directory for plots
import os
OUTPUT_DIR = 'visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*70)
print("Stack Overflow Data Visualization Suite")
print("="*70 + "\n")

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...")
users = load_users()
posts = load_posts()
comments = load_comments()

# Data preprocessing
users['Reputation'] = pd.to_numeric(users['Reputation'], errors='coerce')
users['UpVotes'] = pd.to_numeric(users['UpVotes'], errors='coerce').fillna(0)
users['Views'] = pd.to_numeric(users['Views'], errors='coerce').fillna(0)
users['CreationDate'] = pd.to_datetime(users['CreationDate'], errors='coerce')
users['Year'] = users['CreationDate'].dt.year

posts['PostTypeId'] = pd.to_numeric(posts['PostTypeId'], errors='coerce')
posts['Score'] = pd.to_numeric(posts['Score'], errors='coerce')
posts['ViewCount'] = pd.to_numeric(posts['ViewCount'], errors='coerce')
posts['AnswerCount'] = pd.to_numeric(posts['AnswerCount'], errors='coerce')
posts['OwnerUserId'] = pd.to_numeric(posts['OwnerUserId'], errors='coerce')
posts['CreationDate'] = pd.to_datetime(posts['CreationDate'], errors='coerce')
posts['Year'] = posts['CreationDate'].dt.year
posts['Month'] = posts['CreationDate'].dt.to_period('M')

comments['UserId'] = pd.to_numeric(comments['UserId'], errors='coerce')
comments['Score'] = pd.to_numeric(comments['Score'], errors='coerce')

questions = posts[posts['PostTypeId'] == 1].copy()
answers = posts[posts['PostTypeId'] == 2].copy()

# Extract tags
def extract_tags(tag_string):
    if pd.isna(tag_string) or tag_string == '':
        return []
    return tag_string.replace('><', '|').strip('<>').split('|')

questions['tag_list'] = questions['Tags'].apply(extract_tags)

print("Data loaded successfully!\n")

# =============================================================================
# 1. USER REPUTATION DISTRIBUTION
# =============================================================================

print("Creating visualization 1: User Reputation Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Log-scale histogram
axes[0, 0].hist(users['Reputation'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel('Reputation')
axes[0, 0].set_ylabel('Number of Users (log scale)')
axes[0, 0].set_title('User Reputation Distribution (Log Scale)')
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(users['Reputation'], vert=True)
axes[0, 1].set_ylabel('Reputation')
axes[0, 1].set_title('Reputation Box Plot')
axes[0, 1].grid(True, alpha=0.3)

# CDF
sorted_rep = np.sort(users['Reputation'].dropna())
cdf = np.arange(1, len(sorted_rep) + 1) / len(sorted_rep)
axes[1, 0].plot(sorted_rep, cdf, linewidth=2, color='darkgreen')
axes[1, 0].set_xlabel('Reputation')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Cumulative Distribution of Reputation')
axes[1, 0].set_xscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Top users bar chart
top_10_users = users.nlargest(10, 'Reputation')
axes[1, 1].barh(range(10), top_10_users['Reputation'], color='coral')
axes[1, 1].set_yticks(range(10))
axes[1, 1].set_yticklabels(top_10_users['DisplayName'], fontsize=8)
axes[1, 1].set_xlabel('Reputation')
axes[1, 1].set_title('Top 10 Users by Reputation')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_user_reputation_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/01_user_reputation_analysis.png")
plt.close()

# =============================================================================
# 2. TAG FREQUENCY ANALYSIS
# =============================================================================

print("Creating visualization 2: Tag Frequency Analysis...")

all_tags = [tag.lower().strip() for tags in questions['tag_list'] for tag in tags if tag]
tag_counter = Counter(all_tags)
top_30_tags = tag_counter.most_common(30)

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Top 30 tags bar chart
tags_df = pd.DataFrame(top_30_tags, columns=['Tag', 'Count'])
axes[0].barh(range(len(tags_df)), tags_df['Count'], color='teal')
axes[0].set_yticks(range(len(tags_df)))
axes[0].set_yticklabels(tags_df['Tag'], fontsize=9)
axes[0].set_xlabel('Number of Questions')
axes[0].set_title('Top 30 Most Popular Tags')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Add count labels
for i, count in enumerate(tags_df['Count']):
    axes[0].text(count + 100, i, f'{count:,}', va='center', fontsize=8)

# Tag frequency distribution (power law)
tag_counts = sorted(tag_counter.values(), reverse=True)
axes[1].plot(range(1, len(tag_counts) + 1), tag_counts, linewidth=2, color='darkred')
axes[1].set_xlabel('Tag Rank')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Tag Frequency Distribution (Power Law)')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_tag_frequency_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/02_tag_frequency_analysis.png")
plt.close()

# =============================================================================
# 3. TEMPORAL ACTIVITY PATTERNS
# =============================================================================

print("Creating visualization 3: Temporal Activity Patterns...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Posts by year
posts_by_year = posts.groupby('Year').size()
axes[0, 0].plot(posts_by_year.index, posts_by_year.values, marker='o', linewidth=2, markersize=6, color='purple')
axes[0, 0].fill_between(posts_by_year.index, posts_by_year.values, alpha=0.3, color='purple')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Number of Posts')
axes[0, 0].set_title('Posts Over Time (Annual)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# New users by year
users_by_year = users.groupby('Year').size()
axes[0, 1].plot(users_by_year.index, users_by_year.values, marker='s', linewidth=2, markersize=6, color='green')
axes[0, 1].fill_between(users_by_year.index, users_by_year.values, alpha=0.3, color='green')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('New Users')
axes[0, 1].set_title('User Growth Over Time')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Monthly activity (last 3 years)
recent_posts = posts[posts['CreationDate'] >= '2021-01-01']
posts_by_month = recent_posts.groupby('Month').size()
axes[1, 0].plot(posts_by_month.index.to_timestamp(), posts_by_month.values, linewidth=2, color='orange')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Number of Posts')
axes[1, 0].set_title('Monthly Activity (2021-2024)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Questions vs Answers over time (use posts dataframe directly)
questions_by_year = questions.groupby('Year').size()
answers_by_year = answers.groupby('Year').size()
axes[1, 1].plot(questions_by_year.index, questions_by_year.values, marker='o', label='Questions', linewidth=2)
axes[1, 1].plot(answers_by_year.index, answers_by_year.values, marker='s', label='Answers', linewidth=2)
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Questions vs Answers Over Time')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_temporal_patterns.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/03_temporal_patterns.png")
plt.close()

# =============================================================================
# 4. USER ENGAGEMENT ANALYSIS
# =============================================================================

print("Creating visualization 4: User Engagement Analysis...")

posts_per_user = posts.groupby('OwnerUserId').size()
comments_per_user = comments.groupby('UserId').size()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Posts per user distribution
axes[0, 0].hist(posts_per_user, bins=50, edgecolor='black', alpha=0.7, color='navy')
axes[0, 0].set_xlabel('Number of Posts')
axes[0, 0].set_ylabel('Number of Users')
axes[0, 0].set_title('Distribution of Posts per User')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Comments per user distribution
axes[0, 1].hist(comments_per_user, bins=50, edgecolor='black', alpha=0.7, color='darkred')
axes[0, 1].set_xlabel('Number of Comments')
axes[0, 1].set_ylabel('Number of Users')
axes[0, 1].set_title('Distribution of Comments per User')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# Reputation vs Posts scatter
user_post_counts = posts.groupby('OwnerUserId').size().reset_index(name='PostCount')
users['Id_numeric'] = pd.to_numeric(users['Id'], errors='coerce')
merged = users.merge(user_post_counts, left_on='Id_numeric', right_on='OwnerUserId', how='inner')
sample = merged.sample(min(5000, len(merged)))

axes[1, 0].scatter(sample['PostCount'], sample['Reputation'], alpha=0.5, s=20, color='green')
axes[1, 0].set_xlabel('Number of Posts')
axes[1, 0].set_ylabel('Reputation')
axes[1, 0].set_title('Reputation vs Post Count')
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Top contributors
top_contributors = posts_per_user.nlargest(15)
top_names = []
for user_id in top_contributors.index:
    user_info = users[users['Id_numeric'] == user_id]
    if not user_info.empty:
        top_names.append(user_info.iloc[0]['DisplayName'][:20])
    else:
        top_names.append(f"User {int(user_id)}")

axes[1, 1].barh(range(len(top_contributors)), top_contributors.values, color='skyblue')
axes[1, 1].set_yticks(range(len(top_contributors)))
axes[1, 1].set_yticklabels(top_names, fontsize=8)
axes[1, 1].set_xlabel('Total Posts')
axes[1, 1].set_title('Top 15 Contributors')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_user_engagement.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/04_user_engagement.png")
plt.close()

# =============================================================================
# 5. QUESTION QUALITY METRICS
# =============================================================================

print("Creating visualization 5: Question Quality Metrics...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Score distribution
axes[0, 0].hist(questions['Score'], bins=50, edgecolor='black', alpha=0.7, color='orange', range=(-5, 50))
axes[0, 0].set_xlabel('Question Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Question Score Distribution')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Answer count distribution
axes[0, 1].hist(questions['AnswerCount'], bins=30, edgecolor='black', alpha=0.7, color='purple', range=(0, 15))
axes[0, 1].set_xlabel('Number of Answers')
axes[0, 1].set_ylabel('Number of Questions')
axes[0, 1].set_title('Answer Count Distribution')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# View count distribution
view_counts = questions['ViewCount'].dropna()
view_counts = view_counts[view_counts > 0]
axes[1, 0].hist(view_counts, bins=50, edgecolor='black', alpha=0.7, color='teal')
axes[1, 0].set_xlabel('View Count')
axes[1, 0].set_ylabel('Number of Questions')
axes[1, 0].set_title('Question View Count Distribution')
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Score vs Answer count scatter
sample_questions = questions[['Score', 'AnswerCount']].dropna().sample(min(5000, len(questions)))
axes[1, 1].scatter(sample_questions['AnswerCount'], sample_questions['Score'], alpha=0.3, s=20, color='red')
axes[1, 1].set_xlabel('Number of Answers')
axes[1, 1].set_ylabel('Question Score')
axes[1, 1].set_title('Question Score vs Answer Count')
axes[1, 1].grid(True, alpha=0.3)

# Add correlation
corr = questions[['Score', 'AnswerCount']].corr().iloc[0, 1]
axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 1].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_question_quality.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/05_question_quality.png")
plt.close()

# =============================================================================
# 6. TAG CO-OCCURRENCE HEATMAP
# =============================================================================

print("Creating visualization 6: Tag Co-occurrence Heatmap...")

# Get top 15 tags
top_15_tags = [tag for tag, _ in tag_counter.most_common(15)]

# Build co-occurrence matrix
cooccurrence = pd.DataFrame(0, index=top_15_tags, columns=top_15_tags)

for tags in questions['tag_list']:
    tags_lower = [t.lower() for t in tags if t.lower() in top_15_tags]
    for i, tag1 in enumerate(tags_lower):
        for tag2 in tags_lower[i:]:
            if tag1 != tag2:
                cooccurrence.loc[tag1, tag2] += 1
                cooccurrence.loc[tag2, tag1] += 1

# Create heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd', square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Tag Co-occurrence Matrix (Top 15 Tags)', fontsize=16, pad=20)
plt.xlabel('Tags', fontsize=12)
plt.ylabel('Tags', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_tag_cooccurrence_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/06_tag_cooccurrence_heatmap.png")
plt.close()

# =============================================================================
# 7. USER SPECIALIZATION ANALYSIS
# =============================================================================

print("Creating visualization 7: User Specialization Analysis...")

questions['OwnerUserId'] = pd.to_numeric(questions['OwnerUserId'], errors='coerce')

user_tags_dict = {}
for idx, row in questions.iterrows():
    user_id = row['OwnerUserId']
    tag_list = row['tag_list']
    
    if pd.notna(user_id) and isinstance(tag_list, list) and len(tag_list) > 0:
        if user_id not in user_tags_dict:
            user_tags_dict[user_id] = set()
        user_tags_dict[user_id].update([t.lower() for t in tag_list if t])

tags_per_user = [len(tags) for tags in user_tags_dict.values()]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribution of tags per user
axes[0, 0].hist(tags_per_user, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 0].set_xlabel('Number of Different Tags Used')
axes[0, 0].set_ylabel('Number of Users')
axes[0, 0].set_title('User Tag Diversity Distribution')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Specialists vs Generalists pie chart
specialists = sum(1 for x in tags_per_user if x <= 3)
moderate = sum(1 for x in tags_per_user if 3 < x < 10)
generalists = sum(1 for x in tags_per_user if x >= 10)

labels = ['Specialists\n(≤3 tags)', 'Moderate\n(4-9 tags)', 'Generalists\n(≥10 tags)']
sizes = [specialists, moderate, generalists]
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.1, 0, 0)

axes[0, 1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
axes[0, 1].set_title('User Specialization Categories')

# Cumulative distribution
sorted_tags = np.sort(tags_per_user)
cdf = np.arange(1, len(sorted_tags) + 1) / len(sorted_tags)
axes[1, 0].plot(sorted_tags, cdf, linewidth=2, color='darkblue')
axes[1, 0].set_xlabel('Number of Tags')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Cumulative Distribution of Tag Diversity')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Median')
axes[1, 0].legend()

# Box plot
axes[1, 1].boxplot(tags_per_user, vert=True)
axes[1, 1].set_ylabel('Number of Different Tags')
axes[1, 1].set_title('Tag Diversity Box Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_user_specialization.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/07_user_specialization.png")
plt.close()

# =============================================================================
# 8. CORRELATION MATRIX
# =============================================================================

print("Creating visualization 8: Correlation Matrix...")

# Merge user and post data for correlation analysis
user_metrics = users[['Id_numeric', 'Reputation', 'Views', 'UpVotes']].copy()
user_metrics.columns = ['UserId', 'Reputation', 'ProfileViews', 'UpVotes']

post_metrics = posts.groupby('OwnerUserId').agg({
    'Score': 'mean',
    'ViewCount': 'mean',
    'Id': 'count'
}).reset_index()
post_metrics.columns = ['UserId', 'AvgPostScore', 'AvgPostViews', 'TotalPosts']

comment_metrics = comments.groupby('UserId').size().reset_index(name='TotalComments')

# Merge all
merged_metrics = user_metrics.merge(post_metrics, on='UserId', how='inner')
merged_metrics = merged_metrics.merge(comment_metrics, on='UserId', how='left')
merged_metrics['TotalComments'] = merged_metrics['TotalComments'].fillna(0)

# Select numeric columns for correlation
numeric_cols = ['Reputation', 'ProfileViews', 'UpVotes', 'AvgPostScore', 
                'AvgPostViews', 'TotalPosts', 'TotalComments']
corr_matrix = merged_metrics[numeric_cols].corr()

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of User Metrics', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_correlation_matrix.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/08_correlation_matrix.png")
plt.close()

# =============================================================================
# 9. POWER LAW ANALYSIS
# =============================================================================

print("Creating visualization 9: Power Law Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Reputation power law
sorted_rep = np.sort(users['Reputation'].dropna())[::-1]
axes[0, 0].loglog(range(1, len(sorted_rep) + 1), sorted_rep, linewidth=2, color='blue')
axes[0, 0].set_xlabel('User Rank')
axes[0, 0].set_ylabel('Reputation')
axes[0, 0].set_title('Reputation Power Law Distribution')
axes[0, 0].grid(True, alpha=0.3)

# Post count power law
sorted_posts = np.sort(posts_per_user.values)[::-1]
axes[0, 1].loglog(range(1, len(sorted_posts) + 1), sorted_posts, linewidth=2, color='red')
axes[0, 1].set_xlabel('User Rank')
axes[0, 1].set_ylabel('Number of Posts')
axes[0, 1].set_title('Post Count Power Law Distribution')
axes[0, 1].grid(True, alpha=0.3)

# Tag frequency power law
sorted_tags = sorted(tag_counter.values(), reverse=True)
axes[1, 0].loglog(range(1, len(sorted_tags) + 1), sorted_tags, linewidth=2, color='green')
axes[1, 0].set_xlabel('Tag Rank')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Tag Frequency Power Law')
axes[1, 0].grid(True, alpha=0.3)

# View count power law
sorted_views = np.sort(questions['ViewCount'].dropna())[::-1]
axes[1, 1].loglog(range(1, len(sorted_views) + 1), sorted_views, linewidth=2, color='purple')
axes[1, 1].set_xlabel('Question Rank')
axes[1, 1].set_ylabel('View Count')
axes[1, 1].set_title('Question Views Power Law')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_power_law_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/09_power_law_analysis.png")
plt.close()

# =============================================================================
# 10. STATISTICAL SUMMARY TABLE
# =============================================================================

print("Creating visualization 10: Statistical Summary...")

# Create summary statistics
summary_data = {
    'Metric': [
        'Total Users',
        'Total Questions',
        'Total Answers',
        'Total Comments',
        'Avg Reputation',
        'Median Reputation',
        'Avg Posts/User',
        'Median Posts/User',
        'Avg Question Score',
        'Avg Answer Count',
        'Unanswered Rate',
        'Top 1% Content Share'
    ],
    'Value': [
        f'{len(users):,}',
        f'{len(questions):,}',
        f'{len(answers):,}',
        f'{len(comments):,}',
        f'{users["Reputation"].mean():.2f}',
        f'{users["Reputation"].median():.2f}',
        f'{posts_per_user.mean():.2f}',
        f'{posts_per_user.median():.0f}',
        f'{questions["Score"].mean():.2f}',
        f'{questions["AnswerCount"].mean():.2f}',
        f'{(questions["AnswerCount"] == 0).sum() / len(questions) * 100:.2f}%',
        f'{posts_per_user.nlargest(int(len(posts_per_user) * 0.01)).sum() / len(posts) * 100:.2f}%'
    ]
}

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=[[summary_data['Metric'][i], summary_data['Value'][i]] 
                           for i in range(len(summary_data['Metric']))],
                colLabels=['Metric', 'Value'],
                cellLoc='left',
                loc='center',
                colWidths=[0.6, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style the header
for i in range(2):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data['Metric']) + 1):
    if i % 2 == 0:
        for j in range(2):
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('Stack Overflow Dataset - Statistical Summary', fontsize=16, weight='bold', pad=20)
plt.savefig(f'{OUTPUT_DIR}/10_statistical_summary.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/10_statistical_summary.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("✅ VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
print("\nGenerated visualizations:")
print("  1. User Reputation Analysis (distribution, box plot, CDF, top users)")
print("  2. Tag Frequency Analysis (top tags, power law)")
print("  3. Temporal Activity Patterns (posts/users over time)")
print("  4. User Engagement Analysis (posts/comments distribution, reputation scatter)")
print("  5. Question Quality Metrics (score, answers, views)")
print("  6. Tag Co-occurrence Heatmap (top 15 tags)")
print("  7. User Specialization Analysis (tag diversity)")
print("  8. Correlation Matrix (user metrics)")
print("  9. Power Law Analysis (reputation, posts, tags, views)")
print(" 10. Statistical Summary Table")
print("\n" + "="*70 + "\n")