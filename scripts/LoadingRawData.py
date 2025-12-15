"""
LoadingRawData.py
Simple functions for loading raw XML data from Stack Exchange dump
"""

import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import os

# Configuration
DATA_DIR = 'data/raw'
USERS_FILE = os.path.join(DATA_DIR, 'Users.xml')
POSTS_FILE = os.path.join(DATA_DIR, 'Posts.xml')
COMMENTS_FILE = os.path.join(DATA_DIR, 'Comments.xml')
TAGS_FILE = os.path.join(DATA_DIR, 'Tags.xml')


def load_users():
    """
    Load all users from Users.xml
    
    Returns:
        pd.DataFrame: Users dataframe
    """
    print(f"Loading users from {USERS_FILE}...")
    
    if not os.path.exists(USERS_FILE):
        raise FileNotFoundError(f"File not found: {USERS_FILE}")
    
    users = []
    context = ET.iterparse(USERS_FILE, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in tqdm(context, desc="Parsing Users"):
        if event == 'end' and elem.tag == 'row':
            user = {
                'Id': elem.get('Id'),
                'Reputation': elem.get('Reputation'),
                'DisplayName': elem.get('DisplayName'),
                'CreationDate': elem.get('CreationDate'),
                'LastAccessDate': elem.get('LastAccessDate'),
                'Views': elem.get('Views'),
                'UpVotes': elem.get('UpVotes'),
                'DownVotes': elem.get('DownVotes'),
                'AccountId': elem.get('AccountId'),
                'Location': elem.get('Location'),
                'AboutMe': elem.get('AboutMe')
            }
            users.append(user)
            
            # Clear memory
            elem.clear()
            root.clear()
    
    df = pd.DataFrame(users)
    print(f"✓ Loaded {len(df):,} users")
    return df


def load_posts():
    """
    Load all posts from Posts.xml
    
    Returns:
        pd.DataFrame: Posts dataframe
    """
    print(f"Loading posts from {POSTS_FILE}...")
    
    if not os.path.exists(POSTS_FILE):
        raise FileNotFoundError(f"File not found: {POSTS_FILE}")
    
    posts = []
    context = ET.iterparse(POSTS_FILE, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in tqdm(context, desc="Parsing Posts"):
        if event == 'end' and elem.tag == 'row':
            post = {
                'Id': elem.get('Id'),
                'PostTypeId': elem.get('PostTypeId'),
                'ParentId': elem.get('ParentId'),
                'AcceptedAnswerId': elem.get('AcceptedAnswerId'),
                'CreationDate': elem.get('CreationDate'),
                'Score': elem.get('Score'),
                'ViewCount': elem.get('ViewCount'),
                'Body': elem.get('Body'),
                'OwnerUserId': elem.get('OwnerUserId'),
                'LastEditorUserId': elem.get('LastEditorUserId'),
                'LastEditDate': elem.get('LastEditDate'),
                'LastActivityDate': elem.get('LastActivityDate'),
                'Title': elem.get('Title'),
                'Tags': elem.get('Tags'),
                'AnswerCount': elem.get('AnswerCount'),
                'CommentCount': elem.get('CommentCount'),
                'FavoriteCount': elem.get('FavoriteCount')
            }
            posts.append(post)
            
            elem.clear()
            root.clear()
    
    df = pd.DataFrame(posts)
    print(f"✓ Loaded {len(df):,} posts")
    
    # Show breakdown
    if 'PostTypeId' in df.columns:
        type_counts = df['PostTypeId'].value_counts()
        print(f"  - Questions (type 1): {type_counts.get('1', 0):,}")
        print(f"  - Answers (type 2): {type_counts.get('2', 0):,}")
    
    return df


def load_comments():
    """
    Load all comments from Comments.xml
    
    Returns:
        pd.DataFrame: Comments dataframe
    """
    print(f"Loading comments from {COMMENTS_FILE}...")
    
    if not os.path.exists(COMMENTS_FILE):
        raise FileNotFoundError(f"File not found: {COMMENTS_FILE}")
    
    comments = []
    context = ET.iterparse(COMMENTS_FILE, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in tqdm(context, desc="Parsing Comments"):
        if event == 'end' and elem.tag == 'row':
            comment = {
                'Id': elem.get('Id'),
                'PostId': elem.get('PostId'),
                'Score': elem.get('Score'),
                'Text': elem.get('Text'),
                'CreationDate': elem.get('CreationDate'),
                'UserId': elem.get('UserId'),
                'UserDisplayName': elem.get('UserDisplayName')
            }
            comments.append(comment)
            
            elem.clear()
            root.clear()
    
    df = pd.DataFrame(comments)
    print(f"✓ Loaded {len(df):,} comments")
    return df


def load_tags():
    """
    Load all tags from Tags.xml (if it exists)
    
    Returns:
        pd.DataFrame: Tags dataframe or empty dataframe if file doesn't exist
    """
    if not os.path.exists(TAGS_FILE):
        print(f"⚠️  Tags.xml not found - skipping (tags can be extracted from Posts)")
        return pd.DataFrame(columns=['Id', 'TagName', 'Count', 'ExcerptPostId', 'WikiPostId'])
    
    print(f"Loading tags from {TAGS_FILE}...")
    
    tags = []
    context = ET.iterparse(TAGS_FILE, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in tqdm(context, desc="Parsing Tags"):
        if event == 'end' and elem.tag == 'row':
            tag = {
                'Id': elem.get('Id'),
                'TagName': elem.get('TagName'),
                'Count': elem.get('Count'),
                'ExcerptPostId': elem.get('ExcerptPostId'),
                'WikiPostId': elem.get('WikiPostId')
            }
            tags.append(tag)
            
            elem.clear()
            root.clear()
    
    df = pd.DataFrame(tags)
    print(f"✓ Loaded {len(df):,} tags")
    return df


def get_file_info():
    """Print information about the XML files"""
    from datetime import datetime
    
    files = [
        ('Users.xml', USERS_FILE),
        ('Posts.xml', POSTS_FILE),
        ('Comments.xml', COMMENTS_FILE),
        ('Tags.xml', TAGS_FILE)
    ]
    
    print("\n" + "="*70)
    print("XML File Information")
    print("="*70)
    
    for name, path in files:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            modified = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"{name:20s} | {size_mb:10.2f} MB | Modified: {modified}")
        else:
            print(f"{name:20s} | NOT FOUND")
    
    print("="*70 + "\n")


# Print file info when module is imported
get_file_info()