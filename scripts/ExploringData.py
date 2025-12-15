"""
ExploringData.py
Explore Stack Overflow data directly from XML files to find interesting patterns
No CSV conversion - just pure exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from LoadingRawData import load_users, load_posts, load_comments, load_tags
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "="*70)
print("Stack Overflow Data Exploration - Direct from XML")
print("="*70 + "\n")

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data from XML files...")
print("-" * 70)

users = load_users()
posts = load_posts()
comments = load_comments()
tags = load_tags()  # Will return empty dataframe if file doesn't exist

print("\n" + "="*70)
print("DATA LOADED - Starting Exploration")
print("="*70 + "\n")

# =============================================================================
# BASIC STATISTICS
# =============================================================================

print("="*70)
print("1. BASIC DATASET STATISTICS")
print("="*70)

print(f"\nTotal Users: {len(users):,}")
print(f"Total Posts: {len(posts):,}")
print(f"Total Comments: {len(comments):,}")
if len(tags) > 0:
    print(f"Total Unique Tags (from Tags.xml): {len(tags):,}")
else:
    print(f"Tags.xml not available - will extract from Posts")

# Convert types for analysis
users['Reputation'] = pd.to_numeric(users['Reputation'], errors='coerce')
posts['PostTypeId'] = pd.to_numeric(posts['PostTypeId'], errors='coerce')
posts['Score'] = pd.to_numeric(posts['Score'], errors='coerce')
posts['ViewCount'] = pd.to_numeric(posts['ViewCount'], errors='coerce')
posts['AnswerCount'] = pd.to_numeric(posts['AnswerCount'], errors='coerce')

questions = posts[posts['PostTypeId'] == 1].copy()
answers = posts[posts['PostTypeId'] == 2].copy()

print(f"\nQuestions: {len(questions):,}")
print(f"Answers: {len(answers):,}")
print(f"Questions/Answers Ratio: {len(questions)/len(answers):.2f}")

# =============================================================================
# USER ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("2. USER REPUTATION ANALYSIS")
print("="*70)

rep_stats = users['Reputation'].describe()
print(f"\nReputation Statistics:")
print(f"  Mean: {rep_stats['mean']:.2f}")
print(f"  Median: {rep_stats['50%']:.2f}")
print(f"  Max: {rep_stats['max']:.0f}")
print(f"  Min: {rep_stats['min']:.0f}")

# Power users
high_rep = users[users['Reputation'] > 1000]
super_high_rep = users[users['Reputation'] > 10000]

print(f"\nUsers with >1,000 reputation: {len(high_rep):,} ({len(high_rep)/len(users)*100:.2f}%)")
print(f"Users with >10,000 reputation: {len(super_high_rep):,} ({len(super_high_rep)/len(users)*100:.2f}%)")

# Top 10 users by reputation
users['UpVotes'] = pd.to_numeric(users['UpVotes'], errors='coerce').fillna(0)
users['DownVotes'] = pd.to_numeric(users['DownVotes'], errors='coerce').fillna(0)
users['Views'] = pd.to_numeric(users['Views'], errors='coerce').fillna(0)

top_users = users.nlargest(10, 'Reputation')[['DisplayName', 'Reputation', 'Views', 'UpVotes', 'DownVotes']]
print(f"\n🏆 Top 10 Users by Reputation:")
print(top_users.to_string(index=False))

# =============================================================================
# TAG ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("3. TAG FREQUENCY ANALYSIS")
print("="*70)

# Extract all tags from questions
def extract_tags(tag_string):
    if pd.isna(tag_string) or tag_string == '':
        return []
    return tag_string.replace('><', '|').strip('<>').split('|')

questions['tag_list'] = questions['Tags'].apply(extract_tags)
all_tags = [tag.lower().strip() for tags in questions['tag_list'] for tag in tags if tag]

tag_counter = Counter(all_tags)
top_20_tags = tag_counter.most_common(20)

print(f"\nTotal unique tags in questions: {len(set(all_tags)):,}")
print(f"Total tag occurrences: {len(all_tags):,}")
print(f"Average tags per question: {len(all_tags)/len(questions):.2f}")

print(f"\n📊 Top 20 Most Popular Tags:")
for i, (tag, count) in enumerate(top_20_tags, 1):
    percentage = (count / len(questions)) * 100
    print(f"{i:2d}. {tag:25s} - {count:6,} questions ({percentage:5.2f}%)")

# =============================================================================
# QUESTION ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("4. QUESTION QUALITY METRICS")
print("="*70)

print(f"\nAverage Question Score: {questions['Score'].mean():.2f}")
print(f"Average View Count: {questions['ViewCount'].mean():.0f}")
print(f"Average Answer Count: {questions['AnswerCount'].mean():.2f}")

# Unanswered questions
unanswered = questions[questions['AnswerCount'] == 0]
print(f"\nUnanswered Questions: {len(unanswered):,} ({len(unanswered)/len(questions)*100:.2f}%)")

# High quality questions (high score + many answers)
high_quality = questions[(questions['Score'] >= 10) & (questions['AnswerCount'] >= 5)]
print(f"High Quality Questions (score≥10, answers≥5): {len(high_quality):,}")

# Most viewed questions
top_viewed = questions.nlargest(10, 'ViewCount')[['Title', 'ViewCount', 'Score', 'AnswerCount']]
print(f"\n👁️ Top 10 Most Viewed Questions:")
for idx, row in top_viewed.iterrows():
    title = str(row['Title'])[:60] + "..." if len(str(row['Title'])) > 60 else str(row['Title'])
    print(f"  Views: {row['ViewCount']:7.0f} | Score: {row['Score']:4.0f} | Answers: {row['AnswerCount']:3.0f} | {title}")

# =============================================================================
# ENGAGEMENT PATTERNS
# =============================================================================

print("\n" + "="*70)
print("5. USER ENGAGEMENT PATTERNS")
print("="*70)

posts['OwnerUserId'] = pd.to_numeric(posts['OwnerUserId'], errors='coerce')
comments['UserId'] = pd.to_numeric(comments['UserId'], errors='coerce')

# Posts per user
posts_per_user = posts.groupby('OwnerUserId').size()
print(f"\nAverage posts per user: {posts_per_user.mean():.2f}")
print(f"Median posts per user: {posts_per_user.median():.0f}")
print(f"Max posts by single user: {posts_per_user.max():,}")

# Top contributors
top_contributors = posts_per_user.nlargest(10)
print(f"\n🌟 Top 10 Contributors (by post count):")

users['Id_numeric'] = pd.to_numeric(users['Id'], errors='coerce')

for i, (user_id, post_count) in enumerate(top_contributors.items(), 1):
    user_info = users[users['Id_numeric'] == user_id]
    if not user_info.empty:
        name = user_info.iloc[0]['DisplayName']
        rep = user_info.iloc[0]['Reputation']
        print(f"{i:2d}. User: {name:20s} | Posts: {post_count:5,} | Rep: {rep:,.0f}")
    else:
        print(f"{i:2d}. User ID: {int(user_id)} | Posts: {post_count:5,}")

# Comments per user
comments_per_user = comments.groupby('UserId').size()
print(f"\nAverage comments per user: {comments_per_user.mean():.2f}")
print(f"Total commenting users: {len(comments_per_user):,}")

# =============================================================================
# TAG CO-OCCURRENCE
# =============================================================================

print("\n" + "="*70)
print("6. TAG CO-OCCURRENCE PATTERNS")
print("="*70)

# Find tags that frequently appear together
tag_pairs = []
for tags in questions['tag_list']:
    if len(tags) >= 2:
        for i in range(len(tags)):
            for j in range(i+1, len(tags)):
                pair = tuple(sorted([tags[i].lower(), tags[j].lower()]))
                tag_pairs.append(pair)

pair_counter = Counter(tag_pairs)
top_pairs = pair_counter.most_common(15)

print(f"\n🔗 Top 15 Tag Combinations:")
for i, (pair, count) in enumerate(top_pairs, 1):
    print(f"{i:2d}. {pair[0]:20s} + {pair[1]:20s} = {count:5,} questions")

# =============================================================================
# TEMPORAL PATTERNS
# =============================================================================

print("\n" + "="*70)
print("7. TEMPORAL ACTIVITY PATTERNS")
print("="*70)

posts['CreationDate'] = pd.to_datetime(posts['CreationDate'], errors='coerce')
users['CreationDate'] = pd.to_datetime(users['CreationDate'], errors='coerce')

print(f"\nDataset Time Range:")
print(f"  First post: {posts['CreationDate'].min()}")
print(f"  Last post: {posts['CreationDate'].max()}")
print(f"  Time span: {(posts['CreationDate'].max() - posts['CreationDate'].min()).days} days")

# Posts by year
posts['Year'] = posts['CreationDate'].dt.year
posts_by_year = posts.groupby('Year').size().sort_index()
print(f"\n📅 Posts by Year:")
for year, count in posts_by_year.items():
    if pd.notna(year):
        print(f"  {int(year)}: {count:,} posts")

# New users by year
users['Year'] = users['CreationDate'].dt.year
users_by_year = users.groupby('Year').size().sort_index()
print(f"\n👥 New Users by Year:")
for year, count in users_by_year.items():
    if pd.notna(year):
        print(f"  {int(year)}: {count:,} new users")

# =============================================================================
# INTERESTING FINDINGS
# =============================================================================

print("\n" + "="*70)
print("8. 🔍 INTERESTING FINDINGS")
print("="*70)

# Finding 1: Expertise concentration
posts_per_user_sorted = posts_per_user.sort_values(ascending=False)
top_1_percent = int(len(posts_per_user) * 0.01)
top_contributors_posts = posts_per_user_sorted.head(top_1_percent).sum()
total_posts = len(posts)

print(f"\n💡 Finding 1: Expertise Concentration")
print(f"  Top 1% of users ({top_1_percent:,} users) created {top_contributors_posts:,} posts")
print(f"  That's {(top_contributors_posts/total_posts)*100:.2f}% of all content!")

# Finding 2: Question difficulty (by answer count)
easy_questions = questions[questions['AnswerCount'] >= 3]
hard_questions = questions[(questions['AnswerCount'] == 0) & (questions['ViewCount'] > 100)]

print(f"\n💡 Finding 2: Question Difficulty")
print(f"  'Easy' questions (3+ answers): {len(easy_questions):,} ({len(easy_questions)/len(questions)*100:.2f}%)")
print(f"  'Hard' questions (0 answers, 100+ views): {len(hard_questions):,} ({len(hard_questions)/len(questions)*100:.2f}%)")

# Finding 3: Tag diversity
print(f"\n💡 Finding 3: User Specialization")
print("  Analyzing user tag diversity...")

# Build a mapping of user_id -> set of tags they've used
# Only look at questions since answers don't have tags
questions['OwnerUserId'] = pd.to_numeric(questions['OwnerUserId'], errors='coerce')

user_tags_dict = {}
for idx, row in questions.iterrows():
    user_id = row['OwnerUserId']
    tag_list = row['tag_list']
    
    if pd.notna(user_id) and isinstance(tag_list, list) and len(tag_list) > 0:
        if user_id not in user_tags_dict:
            user_tags_dict[user_id] = set()
        user_tags_dict[user_id].update([t.lower() for t in tag_list if t])

users_with_tags = [len(tags) for tags in user_tags_dict.values()]

if users_with_tags:
    print(f"  Average tags per user: {np.mean(users_with_tags):.2f}")
    print(f"  Median tags per user: {np.median(users_with_tags):.0f}")
    print(f"  Most versatile user uses: {max(users_with_tags)} different tags")
    
    specialists = sum(1 for x in users_with_tags if x <= 3)
    generalists = sum(1 for x in users_with_tags if x >= 10)
    print(f"  Specialists (≤3 tags): {specialists:,} users ({specialists/len(users_with_tags)*100:.2f}%)")
    print(f"  Generalists (≥10 tags): {generalists:,} users ({generalists/len(users_with_tags)*100:.2f}%)")

# Finding 4: Popular vs Niche tags
popular_tags = [tag for tag, count in tag_counter.items() if count >= 100]
niche_tags = [tag for tag, count in tag_counter.items() if count <= 5]

print(f"\n💡 Finding 4: Tag Popularity Distribution")
print(f"  Popular tags (≥100 questions): {len(popular_tags):,}")
print(f"  Niche tags (≤5 questions): {len(niche_tags):,}")
print(f"  Niche tags represent {(len(niche_tags)/len(tag_counter))*100:.2f}% of all tags!")

# Finding 5: User retention over time
print(f"\n💡 Finding 5: Community Growth Pattern")
total_users_by_year = users_by_year.cumsum()
print(f"  Peak growth year: {users_by_year.idxmax():.0f} with {users_by_year.max():,} new users")
print(f"  Recent decline: {users_by_year[2023]} users in 2023 vs {users_by_year[2016]} in 2016")

# =============================================================================
# GRAPH CONSTRUCTION INSIGHTS
# =============================================================================

print("\n" + "="*70)
print("9. 📊 GRAPH CONSTRUCTION INSIGHTS")
print("="*70)

# Potential edges (users who interacted on same question)
print("\nBuilding user interaction preview...")

# Get questions with multiple answers
multi_answer_questions = questions[questions['AnswerCount'] >= 2]
print(f"Questions with 2+ answers: {len(multi_answer_questions):,}")

# Sample calculation for potential edges
answers['ParentId'] = pd.to_numeric(answers['ParentId'], errors='coerce')
questions['Id'] = pd.to_numeric(questions['Id'], errors='coerce')

sample_question_ids = multi_answer_questions['Id'].head(100)
sample_answers = answers[answers['ParentId'].isin(sample_question_ids)]
sample_users = sample_answers['OwnerUserId'].dropna().unique()

print(f"\nSample (first 100 questions with multiple answers):")
print(f"  Involved users: {len(sample_users)}")
if len(sample_users) > 1:
    print(f"  Potential user-user edges: ~{len(sample_users) * (len(sample_users) - 1) / 2:.0f}")

# Estimate for full graph
if len(sample_users) > 0:
    full_estimate = (len(multi_answer_questions) / 100) * len(sample_users)
    print(f"\nFull graph estimate:")
    print(f"  Potential active users in graph: ~{full_estimate:,.0f}")
    print(f"  This will create a dense expert network!")

# Additional insight: Collaborative patterns
print(f"\n🤝 Collaboration Potential:")
questions_with_many_answers = questions[questions['AnswerCount'] >= 5]
print(f"  Questions with 5+ answers: {len(questions_with_many_answers):,}")
print(f"  These create the strongest collaboration clusters")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("✅ EXPLORATION COMPLETE!")
print("="*70)

print("\n🎯 Key Takeaways for Graph Cities Project:")
print("  1. Top 1% of users create majority of content (expertise concentration)")
print("  2. Clear distinction between specialists vs generalists")
print("  3. Strong tag co-occurrence patterns (communities of practice)")
print("  4. Many multi-answer questions → rich user interaction graph")
print("  5. Tags follow power-law: few popular, many niche")
print("  6. Community peaked around 2016-2020, now stabilizing")

print("\n📈 Next Steps:")
print("  1. Build user-user interaction graph from multi-answer questions")
print("  2. Run k-core decomposition to find 'buildings'")
print("  3. Apply TF-IDF to characterize each building's expertise")
print("  4. Use Naive Bayes to predict user communities")
print("  5. Analyze temporal evolution of expert communities")

print("\n" + "="*70 + "\n")