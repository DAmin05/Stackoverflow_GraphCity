"""
CleaningData.py
Simple functions for cleaning and converting Stack Exchange data to CSV
Processes the entire dataset (no sampling)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from LoadingRawData import load_users, load_posts, load_comments, load_tags

# Configuration
OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_tags(tag_string):
    """
    Extract tags from Stack Overflow format '<tag1><tag2><tag3>'
    
    Args:
        tag_string (str): Tag string in SO format
        
    Returns:
        list: List of individual tags
    """
    if pd.isna(tag_string) or tag_string == '':
        return []
    
    # Remove angle brackets and split
    tags = tag_string.replace('><', '|').strip('<>').split('|')
    # Clean and lowercase
    tags = [tag.lower().strip() for tag in tags if tag.strip()]
    return tags


def clean_users(users_df):
    """
    Clean users dataframe
    
    Args:
        users_df (pd.DataFrame): Raw users dataframe
        
    Returns:
        pd.DataFrame: Cleaned users dataframe
    """
    print("\nCleaning users data...")
    df = users_df.copy()
    
    # Convert data types
    df['Id'] = pd.to_numeric(df['Id'], errors='coerce')
    df['Reputation'] = pd.to_numeric(df['Reputation'], errors='coerce')
    df['Views'] = pd.to_numeric(df['Views'], errors='coerce')
    df['UpVotes'] = pd.to_numeric(df['UpVotes'], errors='coerce')
    df['DownVotes'] = pd.to_numeric(df['DownVotes'], errors='coerce')
    df['AccountId'] = pd.to_numeric(df['AccountId'], errors='coerce')
    
    # Convert dates
    df['CreationDate'] = pd.to_datetime(df['CreationDate'], errors='coerce')
    df['LastAccessDate'] = pd.to_datetime(df['LastAccessDate'], errors='coerce')
    
    # Fill missing values
    df['Reputation'] = df['Reputation'].fillna(1)
    df['Views'] = df['Views'].fillna(0)
    df['UpVotes'] = df['UpVotes'].fillna(0)
    df['DownVotes'] = df['DownVotes'].fillna(0)
    
    # Create derived features
    df['TotalVotes'] = df['UpVotes'] + df['DownVotes']
    df['UpVoteRatio'] = df.apply(
        lambda row: row['UpVotes'] / row['TotalVotes'] if row['TotalVotes'] > 0 else 0,
        axis=1
    )
    
    # Clean display names
    df['DisplayName'] = df['DisplayName'].fillna('Anonymous')
    
    # Remove rows with missing essential data
    initial_count = len(df)
    df = df.dropna(subset=['Id'])
    removed = initial_count - len(df)
    
    print(f"  ✓ Cleaned {len(df):,} users (removed {removed:,} with missing IDs)")
    if len(df) > 0:
        print(f"  ✓ Date range: {df['CreationDate'].min()} to {df['CreationDate'].max()}")
        print(f"  ✓ Reputation range: {df['Reputation'].min():.0f} to {df['Reputation'].max():.0f}")
    
    return df


def clean_posts(posts_df):
    """
    Clean posts dataframe
    
    Args:
        posts_df (pd.DataFrame): Raw posts dataframe
        
    Returns:
        pd.DataFrame: Cleaned posts dataframe
    """
    print("\nCleaning posts data...")
    df = posts_df.copy()
    
    # Convert data types
    df['Id'] = pd.to_numeric(df['Id'], errors='coerce')
    df['PostTypeId'] = pd.to_numeric(df['PostTypeId'], errors='coerce')
    df['ParentId'] = pd.to_numeric(df['ParentId'], errors='coerce')
    df['AcceptedAnswerId'] = pd.to_numeric(df['AcceptedAnswerId'], errors='coerce')
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df['ViewCount'] = pd.to_numeric(df['ViewCount'], errors='coerce')
    df['OwnerUserId'] = pd.to_numeric(df['OwnerUserId'], errors='coerce')
    df['AnswerCount'] = pd.to_numeric(df['AnswerCount'], errors='coerce')
    df['CommentCount'] = pd.to_numeric(df['CommentCount'], errors='coerce')
    df['FavoriteCount'] = pd.to_numeric(df['FavoriteCount'], errors='coerce')
    
    # Convert dates
    df['CreationDate'] = pd.to_datetime(df['CreationDate'], errors='coerce')
    df['LastEditDate'] = pd.to_datetime(df['LastEditDate'], errors='coerce')
    df['LastActivityDate'] = pd.to_datetime(df['LastActivityDate'], errors='coerce')
    
    # Fill missing values
    df['Score'] = df['Score'].fillna(0)
    df['ViewCount'] = df['ViewCount'].fillna(0)
    df['AnswerCount'] = df['AnswerCount'].fillna(0)
    df['CommentCount'] = df['CommentCount'].fillna(0)
    df['FavoriteCount'] = df['FavoriteCount'].fillna(0)
    
    # Extract and clean tags
    print("  - Extracting tags...")
    df['TagList'] = df['Tags'].apply(extract_tags)
    df['TagCount'] = df['TagList'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Create flags
    df['IsQuestion'] = (df['PostTypeId'] == 1).astype(int)
    df['IsAnswer'] = (df['PostTypeId'] == 2).astype(int)
    df['HasAcceptedAnswer'] = df['AcceptedAnswerId'].notna().astype(int)
    
    # Remove rows with missing essential data
    initial_count = len(df)
    df = df.dropna(subset=['Id', 'PostTypeId', 'OwnerUserId'])
    removed = initial_count - len(df)
    
    questions_count = df['IsQuestion'].sum()
    answers_count = df['IsAnswer'].sum()
    
    print(f"  ✓ Cleaned {len(df):,} posts (removed {removed:,} with missing data)")
    print(f"  ✓ Questions: {questions_count:,}, Answers: {answers_count:,}")
    if len(df) > 0:
        print(f"  ✓ Date range: {df['CreationDate'].min()} to {df['CreationDate'].max()}")
        print(f"  ✓ Average score: {df['Score'].mean():.2f}")
    
    return df


def clean_comments(comments_df):
    """
    Clean comments dataframe
    
    Args:
        comments_df (pd.DataFrame): Raw comments dataframe
        
    Returns:
        pd.DataFrame: Cleaned comments dataframe
    """
    print("\nCleaning comments data...")
    df = comments_df.copy()
    
    # Convert data types
    df['Id'] = pd.to_numeric(df['Id'], errors='coerce')
    df['PostId'] = pd.to_numeric(df['PostId'], errors='coerce')
    df['UserId'] = pd.to_numeric(df['UserId'], errors='coerce')
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    
    # Convert dates
    df['CreationDate'] = pd.to_datetime(df['CreationDate'], errors='coerce')
    
    # Fill missing values
    df['Score'] = df['Score'].fillna(0)
    df['UserId'] = df['UserId'].fillna(-1)  # -1 for anonymous comments
    df['UserDisplayName'] = df['UserDisplayName'].fillna('Anonymous')
    
    # Remove rows with missing essential data
    initial_count = len(df)
    df = df.dropna(subset=['Id', 'PostId'])
    removed = initial_count - len(df)
    
    print(f"  ✓ Cleaned {len(df):,} comments (removed {removed:,} with missing data)")
    if len(df) > 0:
        print(f"  ✓ Date range: {df['CreationDate'].min()} to {df['CreationDate'].max()}")
        print(f"  ✓ Average score: {df['Score'].mean():.2f}")
    
    return df


def clean_tags(tags_df):
    """
    Clean tags dataframe
    
    Args:
        tags_df (pd.DataFrame): Raw tags dataframe
        
    Returns:
        pd.DataFrame: Cleaned tags dataframe
    """
    print("\nCleaning tags data...")
    df = tags_df.copy()
    
    # Convert data types
    df['Id'] = pd.to_numeric(df['Id'], errors='coerce')
    df['Count'] = pd.to_numeric(df['Count'], errors='coerce')
    df['ExcerptPostId'] = pd.to_numeric(df['ExcerptPostId'], errors='coerce')
    df['WikiPostId'] = pd.to_numeric(df['WikiPostId'], errors='coerce')
    
    # Clean tag names
    df['TagName'] = df['TagName'].str.lower().str.strip()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['TagName'], keep='first')
    
    # Remove rows with missing essential data
    initial_count = len(df)
    df = df.dropna(subset=['Id', 'TagName'])
    removed = initial_count - len(df)
    
    # Sort by count
    df = df.sort_values('Count', ascending=False).reset_index(drop=True)
    
    print(f"  ✓ Cleaned {len(df):,} unique tags (removed {removed:,} duplicates/invalid)")
    if len(df) > 0:
        print(f"  ✓ Total tag occurrences: {df['Count'].sum():,}")
        print(f"  ✓ Most common tag: '{df.iloc[0]['TagName']}' ({df.iloc[0]['Count']:,} uses)")
    
    return df


def save_to_csv(df, filename):
    """
    Save dataframe to CSV
    
    Args:
        df (pd.DataFrame): Dataframe to save
        filename (str): Output filename
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    
    # Get file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  💾 Saved {filename} ({len(df):,} rows, {size_mb:.2f} MB)")


def generate_summary():
    """Generate summary of processed files"""
    print("\n" + "="*70)
    print("Processed Data Summary")
    print("="*70)
    
    files = [
        'users_cleaned.csv',
        'posts_cleaned.csv',
        'comments_cleaned.csv',
        'tags_cleaned.csv'
    ]
    
    for filename in files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            # Count rows efficiently
            with open(filepath, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
            
            # Get file size
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            # Get column count
            df_sample = pd.read_csv(filepath, nrows=1)
            col_count = len(df_sample.columns)
            
            print(f"{filename:25s} | {row_count:10,} rows | {col_count:3d} cols | {size_mb:8.2f} MB")
        else:
            print(f"{filename:25s} | NOT FOUND")
    
    print("="*70 + "\n")


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

print("\n" + "="*70)
print("Stack Overflow Data Processing Pipeline")
print("Processing ENTIRE dataset (no sampling)")
print("="*70)

# Step 1: Load Users
print("\n" + "="*70)
print("STEP 1/4: Loading Users")
print("="*70)
users_raw = load_users()
users_clean = clean_users(users_raw)
save_to_csv(users_clean, 'users_cleaned.csv')

# Step 2: Load and Process Posts
print("\n" + "="*70)
print("STEP 2/4: Loading Posts")
print("="*70)
posts_raw = load_posts()
posts_clean = clean_posts(posts_raw)
save_to_csv(posts_clean, 'posts_cleaned.csv')

# Step 3: Load Comments
print("\n" + "="*70)
print("STEP 3/4: Loading Comments")
print("="*70)
comments_raw = load_comments()
comments_clean = clean_comments(comments_raw)
save_to_csv(comments_clean, 'comments_cleaned.csv')

# Step 4: Load Tags
print("\n" + "="*70)
print("STEP 4/4: Loading Tags")
print("="*70)
tags_raw = load_tags()
tags_clean = clean_tags(tags_raw)
save_to_csv(tags_clean, 'tags_cleaned.csv')

# Final Summary
print("\n" + "="*70)
print("✅ PROCESSING COMPLETE!")
print("="*70)
generate_summary()

print(f"\nAll cleaned CSV files saved to: {OUTPUT_DIR}/")
print("Ready for graph construction and analysis!")
print("="*70 + "\n")