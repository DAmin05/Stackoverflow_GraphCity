"""
Test Building Predictor Model
Tests the trained Naive Bayes classifier and demonstrates predictions
"""

import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
VIZ_DIR = BASE_DIR / "visualizations"

def load_model():
    """Load the trained model"""
    print("Loading trained model...")
    
    with open(MODELS_DIR / "building_predictor.pkl", 'rb') as f:
        model_data = pickle.load(f)
    
    vectorizer = model_data['vectorizer']
    classifier = model_data['classifier']
    
    print(f"✅ Model loaded successfully")
    print(f"   Features: {len(vectorizer.get_feature_names_out())}")
    print(f"   Classes: {len(classifier.classes_)}")
    
    return vectorizer, classifier

def load_building_profiles():
    """Load building profiles for interpretation"""
    with open(ANALYSIS_DIR / "building_tags.json", 'r') as f:
        building_profiles = json.load(f)
    
    return building_profiles

def predict_user_building(user_tags, vectorizer, classifier, building_profiles):
    """
    Predict which building a user belongs to based on their tags
    
    Args:
        user_tags: list of tags, e.g., ['arduino', 'sensors', 'i2c']
        vectorizer: trained TfidfVectorizer
        classifier: trained Naive Bayes classifier
        building_profiles: dict of building data
    
    Returns:
        dict with prediction results
    """
    # Create tag string
    tag_string = ' '.join(user_tags)
    
    # Vectorize
    X_vec = vectorizer.transform([tag_string])
    
    # Predict
    prediction = classifier.predict(X_vec)[0]
    probabilities = classifier.predict_proba(X_vec)[0]
    
    # Get building names and probabilities
    buildings = classifier.classes_
    prob_dict = {building: float(prob) for building, prob in zip(buildings, probabilities)}
    
    # Sort by probability
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Get building info
    k_core = int(prediction.split('_')[1])
    profile = building_profiles.get(prediction, {})
    
    result = {
        'input_tags': user_tags,
        'predicted_building': prediction,
        'k_core_level': k_core,
        'confidence': sorted_probs[0][1],
        'primary_domain': profile.get('primary_domain', 'Unknown'),
        'building_top_tags': profile.get('top_5_tags', []),
        'top_5_predictions': dict(sorted_probs[:5])
    }
    
    return result

def test_on_sample_users(vectorizer, classifier, building_profiles):
    """Test predictions on sample user profiles"""
    print("\n" + "="*80)
    print("TESTING MODEL ON SAMPLE USERS")
    print("="*80)
    
    # Define test cases representing different expertise levels
    test_cases = [
        {
            'name': 'Arduino Beginner',
            'tags': ['arduino', 'led', 'resistor', 'programming', 'digitalWrite'],
            'expected_level': 'Beginner to Intermediate'
        },
        {
            'name': 'Arduino Intermediate',
            'tags': ['arduino', 'serial', 'sensors', 'i2c', 'library', 'interrupt'],
            'expected_level': 'Intermediate'
        },
        {
            'name': 'Embedded Systems Expert',
            'tags': ['microcontroller', 'embedded', 'c', 'arm', 'cortex', 'rtos', 'bare-metal'],
            'expected_level': 'Expert'
        },
        {
            'name': 'RF/Wireless Expert',
            'tags': ['rf', 'antenna', 'impedance-matching', 'smith-chart', 'pcb-design', 'transmission-line'],
            'expected_level': 'Expert'
        },
        {
            'name': 'Power Electronics Specialist',
            'tags': ['power-supply', 'buck-converter', 'mosfet', 'switching-regulator', 'pwm', 'efficiency'],
            'expected_level': 'Advanced to Expert'
        },
        {
            'name': 'FPGA Developer',
            'tags': ['fpga', 'vhdl', 'xilinx', 'digital-logic', 'timing', 'synthesis'],
            'expected_level': 'Expert'
        },
        {
            'name': 'Raspberry Pi Hobbyist',
            'tags': ['raspberry-pi', 'python', 'gpio', 'i2c', 'sensors', 'linux'],
            'expected_level': 'Intermediate'
        },
        {
            'name': 'PCB Design Engineer',
            'tags': ['pcb-design', 'eagle', 'kicad', 'layout', 'routing', 'gerber', 'vias'],
            'expected_level': 'Advanced to Expert'
        },
        {
            'name': 'Analog Circuit Designer',
            'tags': ['operational-amplifier', 'opamp', 'filter', 'analog', 'circuit-analysis', 'frequency-response'],
            'expected_level': 'Expert'
        },
        {
            'name': 'Electronics Novice',
            'tags': ['batteries', 'voltage', 'current', 'led', 'resistor'],
            'expected_level': 'Beginner'
        }
    ]
    
    results = []
    
    print("\n🧪 Running predictions...\n")
    
    for test_case in test_cases:
        result = predict_user_building(test_case['tags'], vectorizer, classifier, building_profiles)
        result['name'] = test_case['name']
        result['expected_level'] = test_case['expected_level']
        results.append(result)
        
        # Display result
        print(f"👤 {test_case['name']}")
        print(f"   Expected Level: {test_case['expected_level']}")
        print(f"   Input Tags: {', '.join(test_case['tags'])}")
        print(f"   → Predicted Building: {result['predicted_building']}")
        print(f"   → K-Core Level: {result['k_core_level']}")
        print(f"   → Primary Domain: {result['primary_domain']}")
        print(f"   → Confidence: {result['confidence']:.1%}")
        print(f"   → Building's Top Tags: {', '.join(result['building_top_tags'][:5])}")
        print(f"   → Top 3 Predictions:")
        for building, prob in list(result['top_5_predictions'].items())[:3]:
            k_core = int(building.split('_')[1])
            print(f"      {building} (k-core={k_core}): {prob:.1%}")
        print()
    
    return results

def evaluate_model_performance(vectorizer, classifier, building_profiles):
    """Evaluate model on known user data"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*80)
    
    # Load user data
    user_df = pd.read_csv(ANALYSIS_DIR / "user_building_assignments.csv")
    
    print(f"\nEvaluating on {len(user_df)} users...")
    
    # Load building tags to reconstruct user tag profiles
    with open(ANALYSIS_DIR / "building_tags.json", 'r') as f:
        building_tags = json.load(f)
    
    # Create test samples (subsample for speed)
    sample_size = min(1000, len(user_df))
    user_sample = user_df.sample(n=sample_size, random_state=42)
    
    X_test = []
    y_test = []
    
    print(f"Creating test samples from {sample_size} users...")
    
    for _, user in user_sample.iterrows():
        building = user['building']
        
        # Get tags for this building
        if building in building_tags:
            tags = building_tags[building].get('top_10_tags', [])
            if tags:
                # Create tag string
                tag_string = ' '.join(tags[:5])  # Use 5 random tags
                X_test.append(tag_string)
                y_test.append(building)
    
    print(f"Created {len(X_test)} test samples")
    
    # Vectorize
    X_test_vec = vectorizer.transform(X_test)
    
    # Predict
    y_pred = classifier.predict(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 ACCURACY: {accuracy:.2%}")
    
    # Classification report
    print("\n📋 DETAILED METRICS:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix for top buildings
    unique_buildings = sorted(set(y_test) | set(y_pred))
    top_buildings = unique_buildings[:10]
    
    # Filter to top buildings only
    mask = [y in top_buildings for y in y_test]
    y_test_top = [y for y, m in zip(y_test, mask) if m]
    y_pred_top = [y for y, m in zip(y_pred, mask) if m]
    
    if len(y_test_top) > 0:
        cm = confusion_matrix(y_test_top, y_pred_top, labels=top_buildings)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[b.replace('wavemap_', '') for b in top_buildings],
                    yticklabels=[b.replace('wavemap_', '') for b in top_buildings])
        plt.title('Confusion Matrix (Top 10 Buildings)', fontsize=14)
        plt.xlabel('Predicted Building', fontsize=12)
        plt.ylabel('True Building', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "model_confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved confusion matrix to {VIZ_DIR / 'model_confusion_matrix.png'}")
        plt.close()

def interactive_mode(vectorizer, classifier, building_profiles):
    """Interactive prediction mode"""
    print("\n" + "="*80)
    print("INTERACTIVE PREDICTION MODE")
    print("="*80)
    print("\nEnter tags separated by spaces (or 'quit' to exit)")
    print("Example: arduino sensors i2c programming\n")
    
    while True:
        user_input = input("Enter tags: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        tags = user_input.split()
        result = predict_user_building(tags, vectorizer, classifier, building_profiles)
        
        print(f"\n🔍 Prediction Results:")
        print(f"   Building: {result['predicted_building']}")
        print(f"   K-Core Level: {result['k_core_level']}")
        print(f"   Primary Domain: {result['primary_domain']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"\n   Top 3 Predictions:")
        for building, prob in list(result['top_5_predictions'].items())[:3]:
            print(f"      {building}: {prob:.1%}")
        print()

def analyze_feature_importance(vectorizer, classifier, building_profiles):
    """Analyze which tags are most important for each building"""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top features for each class
    print("\n📊 Most Important Tags Per Building:\n")
    
    # Sort buildings by k-core
    sorted_buildings = sorted(classifier.classes_, 
                             key=lambda x: int(x.split('_')[1]), 
                             reverse=True)
    
    for building in sorted_buildings[:5]:  # Top 5 buildings
        # Get feature log probabilities for this class
        class_idx = list(classifier.classes_).index(building)
        feature_probs = classifier.feature_log_prob_[class_idx]
        
        # Get top features
        top_indices = np.argsort(feature_probs)[-10:][::-1]
        top_features = [(feature_names[i], feature_probs[i]) for i in top_indices]
        
        k_core = building.split('_')[1]
        profile = building_profiles.get(building, {})
        
        print(f"📍 {building} (k-core={k_core})")
        print(f"   Primary Domain: {profile.get('primary_domain', 'Unknown')}")
        print(f"   Most Important Tags:")
        for tag, prob in top_features:
            print(f"      • {tag}")
        print()

def main():
    """Main execution"""
    print("="*80)
    print("BUILDING PREDICTOR MODEL - TESTING & EVALUATION")
    print("="*80)
    
    # Load model
    vectorizer, classifier = load_model()
    building_profiles = load_building_profiles()
    
    # Test on sample users
    results = test_on_sample_users(vectorizer, classifier, building_profiles)
    
    # Evaluate performance
    evaluate_model_performance(vectorizer, classifier, building_profiles)
    
    # Analyze feature importance
    analyze_feature_importance(vectorizer, classifier, building_profiles)
    
    # Interactive mode (optional)
    print("\n" + "="*80)
    response = input("Would you like to try interactive prediction mode? (y/n): ")
    if response.lower() in ['y', 'yes']:
        interactive_mode(vectorizer, classifier, building_profiles)
    
    print("\n✅ Testing complete!")
    print(f"Visualizations saved to: {VIZ_DIR}")

if __name__ == '__main__':
    main()