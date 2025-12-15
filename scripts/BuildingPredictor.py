"""
Building Predictor - Naive Bayes Classifier
Predicts which building (community) a user belongs to based on their tags
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
VIZ_DIR = BASE_DIR / "visualizations"
MODELS_DIR = BASE_DIR / "models"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

def load_data():
    """Load building tags and user assignments"""
    print("Loading data...")
    
    # Load building profiles
    with open(ANALYSIS_DIR / "building_tags.json", 'r') as f:
        building_profiles = json.load(f)
    
    # Load user assignments
    user_df = pd.read_csv(ANALYSIS_DIR / "user_building_assignments.csv")
    
    print(f"Loaded {len(building_profiles)} buildings")
    print(f"Loaded {len(user_df)} users")
    
    return building_profiles, user_df

def prepare_training_data(building_profiles, user_df):
    """
    Prepare training data for classification
    Each user's tags become a training sample
    """
    print("\nPreparing training data...")
    
    # For this version, we'll create synthetic training data from building profiles
    # In a real scenario, you'd extract actual user tags
    
    X = []  # Tag strings
    y = []  # Building labels
    
    # Generate training samples from building profiles
    for building, profile in building_profiles.items():
        # Skip buildings with very few users
        if profile['user_count'] < 5:
            continue
        
        # Create multiple samples per building based on tag distribution
        tags = profile['top_20_tags']
        tag_counts = profile['tag_counts']
        
        # Generate samples (proportional to user count)
        num_samples = min(profile['user_count'], 100)  # Cap at 100 samples per building
        
        for _ in range(num_samples):
            # Randomly sample tags weighted by their frequency
            sample_tags = np.random.choice(
                list(tag_counts.keys()),
                size=min(10, len(tags)),
                p=[tag_counts[t]/sum(tag_counts.values()) for t in tag_counts.keys()],
                replace=False
            )
            
            # Create tag string
            tag_string = ' '.join(sample_tags)
            X.append(tag_string)
            y.append(building)
    
    print(f"Created {len(X)} training samples")
    print(f"Classes: {len(set(y))} buildings")
    
    return X, y

def train_classifier(X, y):
    """Train Naive Bayes classifier"""
    print("\nTraining classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Vectorize
    print("\nVectorizing tags with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=2,
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix: {X_train_vec.shape}")
    
    # Train Naive Bayes
    print("\nTraining Naive Bayes classifier...")
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy:.3f}")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Detailed metrics
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return vectorizer, clf, (X_test, y_test, y_pred)

def save_model(vectorizer, clf):
    """Save trained model"""
    model_data = {
        'vectorizer': vectorizer,
        'classifier': clf
    }
    
    with open(MODELS_DIR / "building_predictor.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to {MODELS_DIR / 'building_predictor.pkl'}")

def visualize_results(y_test, y_pred, building_profiles):
    """Create visualizations"""
    print("\nCreating visualizations...")
    
    # Get unique buildings
    buildings = sorted(set(y_test) | set(y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=buildings)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Confusion Matrix (top buildings only for readability)
    top_buildings = buildings[:15]
    cm_subset = confusion_matrix(y_test, y_pred, labels=top_buildings)
    
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=[b.replace('wavemap_', '') for b in top_buildings],
                yticklabels=[b.replace('wavemap_', '') for b in top_buildings])
    ax1.set_title('Confusion Matrix (Top 15 Buildings)', fontsize=14)
    ax1.set_xlabel('Predicted Building', fontsize=12)
    ax1.set_ylabel('True Building', fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Accuracy by building size
    building_sizes = {b: building_profiles[b]['user_count'] for b in buildings}
    building_accuracies = {}
    
    for building in buildings:
        true_pos = cm[buildings.index(building), buildings.index(building)]
        total = cm[buildings.index(building), :].sum()
        acc = true_pos / total if total > 0 else 0
        building_accuracies[building] = acc
    
    df_plot = pd.DataFrame({
        'building': list(building_sizes.keys()),
        'user_count': list(building_sizes.values()),
        'accuracy': [building_accuracies[b] for b in building_sizes.keys()]
    })
    
    ax2.scatter(df_plot['user_count'], df_plot['accuracy'], s=100, alpha=0.6)
    ax2.set_xlabel('Building Size (User Count)', fontsize=12)
    ax2.set_ylabel('Classification Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Building Size', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "building_prediction_results.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {VIZ_DIR / 'building_prediction_results.png'}")
    
    plt.close()

def predict_building(user_tags, vectorizer, clf):
    """
    Predict which building a user belongs to
    
    Args:
        user_tags: list of tags, e.g., ['arduino', 'sensors', 'i2c']
        vectorizer: trained TfidfVectorizer
        clf: trained classifier
    
    Returns:
        prediction: predicted building name
        probabilities: dict of building -> probability (top 5)
    """
    # Create tag string
    tag_string = ' '.join(user_tags)
    
    # Vectorize
    X_vec = vectorizer.transform([tag_string])
    
    # Predict
    prediction = clf.predict(X_vec)[0]
    probabilities = clf.predict_proba(X_vec)[0]
    
    # Get building names and probabilities
    buildings = clf.classes_
    prob_dict = {building: float(prob) for building, prob in zip(buildings, probabilities)}
    
    # Sort by probability
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    return prediction, dict(sorted_probs[:5])

def interactive_prediction(vectorizer, clf, building_profiles):
    """Interactive prediction mode"""
    print("\n" + "="*80)
    print("INTERACTIVE BUILDING PREDICTION")
    print("="*80)
    
    # Test cases
    test_cases = [
        {
            'name': 'Arduino Beginner',
            'tags': ['arduino', 'led', 'resistor', 'programming']
        },
        {
            'name': 'RF Expert',
            'tags': ['rf', 'antenna', 'impedance-matching', 'smith-chart', 'pcb-design']
        },
        {
            'name': 'Power Electronics Specialist',
            'tags': ['power-supply', 'buck-converter', 'mosfet', 'switching-regulator']
        },
        {
            'name': 'FPGA Developer',
            'tags': ['fpga', 'vhdl', 'xilinx', 'digital-logic', 'timing']
        },
        {
            'name': 'Raspberry Pi Hobbyist',
            'tags': ['raspberry-pi', 'python', 'gpio', 'i2c', 'sensors']
        }
    ]
    
    print("\n🧪 Testing predictions on sample users:\n")
    
    for test_case in test_cases:
        prediction, probabilities = predict_building(test_case['tags'], vectorizer, clf)
        
        k_core = int(prediction.split('_')[1])
        profile = building_profiles[prediction]
        
        print(f"👤 {test_case['name']}")
        print(f"   Tags: {', '.join(test_case['tags'])}")
        print(f"   → Predicted Building: {prediction}")
        print(f"   → K-Core Level: {k_core}")
        print(f"   → Primary Domain: {profile['primary_domain']}")
        print(f"   → Confidence: {probabilities[prediction]:.1%}")
        print(f"   → Top 3 Predictions:")
        for building, prob in list(probabilities.items())[:3]:
            print(f"      {building}: {prob:.1%}")
        print()

def main():
    """Main execution"""
    print("="*80)
    print("BUILDING PREDICTOR - NAIVE BAYES CLASSIFIER")
    print("="*80)
    
    # Load data
    building_profiles, user_df = load_data()
    
    # Prepare training data
    X, y = prepare_training_data(building_profiles, user_df)
    
    # Train classifier
    vectorizer, clf, test_results = train_classifier(X, y)
    
    # Save model
    save_model(vectorizer, clf)
    
    # Visualize
    X_test, y_test, y_pred = test_results
    visualize_results(y_test, y_pred, building_profiles)
    
    # Interactive predictions
    interactive_prediction(vectorizer, clf, building_profiles)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {MODELS_DIR}")
    print(f"Visualizations saved to: {VIZ_DIR}")

if __name__ == '__main__':
    main()