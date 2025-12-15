"""
Building Tag Extractor
Extracts and analyzes tag distributions for each building in the Graph City
Maps expertise domains to communities
"""

import pandas as pd
import json
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent.parent
GRAPH_DATA_DIR = BASE_DIR / "graph_data"
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "analysis"

# Create directories
ANALYSIS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load all necessary data"""
    print("Loading data...")
    
    # Load user k-cores (must run QuickPrediction.py first)
    user_df = pd.read_csv(ANALYSIS_DIR / "user_k_cores.csv")
    
    # Load original labels - NO HEADER!
    labels_df = pd.read_csv(
        GRAPH_DATA_DIR / "stackoverflow_electronics_main_labels.csv",
        header=None,  # Important: no header row!
        names=['original_user_id', 'username']
    )
    
    print(f"User DF columns: {list(user_df.columns)}")
    print(f"Labels DF shape: {labels_df.shape}")
    print(f"Labels DF sample:")
    print(labels_df.head())
    
    # Add reindexed user_id (row index = reindexed ID)
    labels_df['user_id'] = labels_df.index
    
    # Merge with user_df
    user_df = user_df.merge(
        labels_df[['user_id', 'original_user_id', 'username']], 
        on='user_id', 
        how='left'
    )
    
    print(f"\n✅ Loaded {len(user_df)} users")
    print(f"✅ Mapped {user_df['original_user_id'].notna().sum()} users to original IDs")
    print(f"\nSample data:")
    print(user_df[['user_id', 'original_user_id', 'username', 'k_core']].head())
    
    return user_df

def assign_buildings(user_df):
    """Assign users to specific buildings based on k-core"""
    def get_primary_building(k_core):
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
            return "wavemap_5_12699_1"
        elif k_core >= 4:
            return "wavemap_4_179_1"
        elif k_core >= 3:
            return "wavemap_3_17881_1"
        elif k_core >= 2:
            return "wavemap_2_16614_1"
        else:
            return "wavemap_1_17795_20"
    
    user_df['building'] = user_df['k_core'].apply(get_primary_building)
    print(f"\n✅ Assigned {len(user_df)} users to buildings")
    return user_df

def extract_user_tags():
    """Extract tags from Posts.xml for each user"""
    print("\nExtracting tags from Posts.xml...")
    print("(This may take a few minutes...)")
    
    user_tags = defaultdict(Counter)
    
    posts_file = DATA_DIR / "raw" / "Posts.xml"
    
    if not posts_file.exists():
        print(f"⚠️  Warning: {posts_file} not found!")
        return user_tags
    
    try:
        context = ET.iterparse(posts_file, events=('end',))
        
        count = 0
        for event, elem in context:
            if elem.tag == 'row':
                count += 1
                if count % 50000 == 0:
                    print(f"  Processed {count:,} posts...")
                
                owner_id = elem.get('OwnerUserId')
                tags = elem.get('Tags')
                
                if owner_id and tags:
                    try:
                        owner_id = int(owner_id)
                        # Parse tags: <arduino><sensors><i2c> -> ['arduino', 'sensors', 'i2c']
                        tag_list = tags.replace('><', ',').strip('<>').split(',')
                        for tag in tag_list:
                            if tag:
                                user_tags[owner_id][tag] += 1
                    except ValueError:
                        pass
                
                elem.clear()
        
        print(f"\n✅ Processed {count:,} posts")
        print(f"✅ Found tags for {len(user_tags):,} users")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return user_tags

def aggregate_building_tags(user_df, user_tags):
    """Aggregate tags for each building"""
    print("\nAggregating tags per building...")
    
    building_tags = defaultdict(Counter)
    building_user_counts = defaultdict(int)
    
    matched_users = 0
    
    for _, user in user_df.iterrows():
        user_id = user['user_id']
        building = user['building']
        
        building_user_counts[building] += 1
        
        # Use original Stack Overflow user ID to match with Posts.xml
        if pd.notna(user.get('original_user_id')):
            orig_id = int(user['original_user_id'])
            tags = user_tags.get(orig_id)
            
            if tags:
                matched_users += 1
                building_tags[building].update(tags)
    
    buildings_with_tags = len([b for b in building_tags.values() if len(b) > 0])
    print(f"✅ Matched {matched_users:,} users to their tags")
    print(f"✅ Found tags for {buildings_with_tags}/{len(building_user_counts)} buildings")
    
    return building_tags, building_user_counts

def create_building_profiles(building_tags, building_user_counts):
    """Create detailed profiles for each building"""
    print("\nCreating building profiles...")
    
    building_profiles = {}
    
    for building, tag_counter in building_tags.items():
        if len(tag_counter) == 0:
            continue
            
        # Get top tags
        top_20 = tag_counter.most_common(20)
        top_10 = tag_counter.most_common(10)
        top_5 = tag_counter.most_common(5)
        
        # Calculate tag statistics
        total_tags = sum(tag_counter.values())
        unique_tags = len(tag_counter)
        
        # Calculate diversity (entropy)
        from math import log
        diversity = 0
        if total_tags > 0:
            for count in tag_counter.values():
                p = count / total_tags
                if p > 0:
                    diversity -= p * log(p)
        
        building_profiles[building] = {
            'user_count': building_user_counts[building],
            'total_tags': total_tags,
            'unique_tags': unique_tags,
            'diversity': round(diversity, 3),
            'top_5_tags': [tag for tag, _ in top_5],
            'top_10_tags': [tag for tag, _ in top_10],
            'top_20_tags': [tag for tag, _ in top_20],
            'tag_counts': dict(top_20),
            'avg_tags_per_user': round(total_tags / building_user_counts[building], 2) if building_user_counts[building] > 0 else 0
        }
    
    print(f"✅ Created profiles for {len(building_profiles)} buildings")
    return building_profiles

def analyze_building_expertise(building_profiles):
    """Analyze expertise domains for each building"""
    print("\nAnalyzing building expertise domains...")
    
    # Common expertise domains in electronics
    domain_keywords = {
        'Arduino/Microcontrollers': ['arduino', 'atmega', 'pic', 'avr', 'microcontroller', 'embedded'],
        'Raspberry Pi/Computing': ['raspberry-pi', 'linux', 'gpio', 'python', 'spi', 'i2c'],
        'Power Electronics': ['power-supply', 'voltage-regulator', 'buck-converter', 'transformer', 'battery'],
        'RF/Wireless': ['rf', 'antenna', 'wifi', 'bluetooth', 'wireless', 'radio'],
        'Analog Circuits': ['operational-amplifier', 'opamp', 'analog', 'filter', 'amplifier'],
        'Digital Logic': ['fpga', 'vhdl', 'verilog', 'digital-logic', 'logic-gates'],
        'PCB Design': ['pcb', 'pcb-design', 'layout', 'eagle', 'kicad'],
        'Sensors/Measurement': ['sensor', 'adc', 'measurement', 'temperature', 'pressure'],
        'Motors/Actuators': ['motor', 'stepper-motor', 'servo', 'driver', 'h-bridge'],
        'Audio/Video': ['audio', 'amplifier', 'speaker', 'video', 'display']
    }
    
    for building, profile in building_profiles.items():
        domains = {}
        top_tags = profile['top_20_tags']
        
        for domain, keywords in domain_keywords.items():
            # Count how many domain keywords appear in top tags
            matches = sum(1 for tag in top_tags if any(kw in tag for kw in keywords))
            if matches > 0:
                domains[domain] = matches
        
        # Sort domains by relevance
        profile['expertise_domains'] = sorted(domains.items(), key=lambda x: x[1], reverse=True)
        profile['primary_domain'] = profile['expertise_domains'][0][0] if profile['expertise_domains'] else 'General'
    
    return building_profiles

def save_results(building_profiles, user_df):
    """Save all results"""
    # Save building profiles
    with open(ANALYSIS_DIR / "building_tags.json", 'w') as f:
        json.dump(building_profiles, f, indent=2)
    
    print(f"\n✅ Saved building profiles to {ANALYSIS_DIR / 'building_tags.json'}")
    
    # Save user-building assignments
    user_df[['user_id', 'original_user_id', 'username', 'k_core', 'building', 'expertise_level']].to_csv(
        ANALYSIS_DIR / "user_building_assignments.csv", 
        index=False
    )
    
    print(f"✅ Saved user assignments to {ANALYSIS_DIR / 'user_building_assignments.csv'}")
    
    # Create summary
    summary = {
        'total_buildings': len(building_profiles),
        'buildings': list(building_profiles.keys()),
        'building_summaries': {
            building: {
                'users': profile['user_count'],
                'top_5_tags': profile['top_5_tags'],
                'primary_domain': profile['primary_domain']
            }
            for building, profile in building_profiles.items()
        }
    }
    
    with open(ANALYSIS_DIR / "building_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved summary to {ANALYSIS_DIR / 'building_summary.json'}")

def display_results(building_profiles):
    """Display key findings"""
    print("\n" + "="*80)
    print("BUILDING TAG ANALYSIS RESULTS")
    print("="*80)
    
    # Sort buildings by k-core (from name)
    sorted_buildings = sorted(building_profiles.items(), 
                             key=lambda x: int(x[0].split('_')[1]), 
                             reverse=True)
    
    print("\n🏢 BUILDING EXPERTISE PROFILES:\n")
    
    for building, profile in sorted_buildings[:10]:  # Top 10 buildings
        k_core = building.split('_')[1]
        print(f"\n📍 {building} (k-core={k_core})")
        print(f"   Users: {profile['user_count']}")
        print(f"   Primary Domain: {profile['primary_domain']}")
        print(f"   Top 5 Tags: {', '.join(profile['top_5_tags'])}")
        print(f"   Tag Diversity: {profile['diversity']:.2f}")
    
    print("\n" + "="*80)
    
    # Domain distribution across all buildings
    print("\n🎯 EXPERTISE DOMAIN DISTRIBUTION:\n")
    all_domains = defaultdict(int)
    for profile in building_profiles.values():
        primary = profile['primary_domain']
        all_domains[primary] += 1
    
    for domain, count in sorted(all_domains.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {count} buildings")
    
    print("\n" + "="*80)

def main():
    """Main execution"""
    print("="*80)
    print("BUILDING TAG EXTRACTOR")
    print("="*80)
    
    # Load data
    user_df = load_data()
    
    # Assign buildings
    user_df = assign_buildings(user_df)
    
    # Extract tags
    user_tags = extract_user_tags()
    
    # Aggregate by building
    building_tags, building_user_counts = aggregate_building_tags(user_df, user_tags)
    
    # Create profiles
    building_profiles = create_building_profiles(building_tags, building_user_counts)
    
    # Analyze expertise domains
    building_profiles = analyze_building_expertise(building_profiles)
    
    # Save results
    save_results(building_profiles, user_df)
    
    # Display results
    display_results(building_profiles)
    
    print("\n✅ Tag extraction complete!")
    print(f"Results saved to: {ANALYSIS_DIR}")

if __name__ == '__main__':
    main()