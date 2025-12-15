"""
Stack Overflow Electronics Graph City Dashboard
Complete visualization and analysis dashboard
"""

import streamlit as st
import pandas as pd
import json
import pickle
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import subprocess
import sys
import time

# Page config
st.set_page_config(
    page_title="Stack Overflow Electronics Graph City",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define paths (Fixed with .resolve() for absolute paths)
BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
MODELS_DIR = BASE_DIR / "models"
GRAPH_DATA_DIR = BASE_DIR / "graph_data"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Custom CSS - Better colors and visibility
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #58a6ff;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #c9d1d9;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #58a6ff;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #58a6ff;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        font-size: 1rem;
        color: #e0e0e0;
    }
    .success-box {
        background-color: #1a4d2e;
        border-left: 5px solid #4caf50;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #c9d1d9;
    }
    .error-box {
        background-color: #4d1a1a;
        border-left: 5px solid #f44336;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #c9d1d9;
    }
    .warning-box {
        background-color: #4d3a1a;
        border-left: 5px solid #ff9800;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #c9d1d9;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #58a6ff !important;
    }
    .stDataFrame {
        color: #c9d1d9;
    }
    div[data-testid="stMetricValue"] {
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

def run_script(script_name, description):
    """Run a Python script and show progress"""
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return False
    
    st.info(f"🔄 Running {description}...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Run the script
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPTS_DIR)
        )
        
        # Simulate progress (since we can't track actual progress easily)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
            if i % 20 == 0:
                status_text.text(f"Processing... {i+1}%")
            
            # Check if process finished
            if process.poll() is not None:
                break
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            st.success(f"✅ {description} completed successfully!")
            return True
        else:
            st.error(f"❌ Error running {description}")
            with st.expander("Show error details"):
                st.code(stderr)
            return False
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False

# Cache data loading
@st.cache_data
def load_all_data():
    """Load all necessary data"""
    try:
        required_files = {
            'user_k_cores.csv': ANALYSIS_DIR / "user_k_cores.csv",
            'building_tags.json': ANALYSIS_DIR / "building_tags.json",
            'building_importance.csv': ANALYSIS_DIR / "building_importance.csv",
            'user_building_assignments.csv': ANALYSIS_DIR / "user_building_assignments.csv",
            'k_core_summary.json': ANALYSIS_DIR / "k_core_summary.json",
            'stackoverflow_electronics_main.txt': GRAPH_DATA_DIR / "stackoverflow_electronics_main.txt"
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(name)
        
        if missing_files:
            return None, None, None, None, None, None, missing_files
        
        # Load all data
        user_df = pd.read_csv(ANALYSIS_DIR / "user_k_cores.csv")
        
        with open(ANALYSIS_DIR / "building_tags.json", 'r') as f:
            building_profiles = json.load(f)
        
        importance_df = pd.read_csv(ANALYSIS_DIR / "building_importance.csv")
        user_assignments = pd.read_csv(ANALYSIS_DIR / "user_building_assignments.csv")
        
        with open(ANALYSIS_DIR / "k_core_summary.json", 'r') as f:
            summary = json.load(f)
        
        edges_df = pd.read_csv(GRAPH_DATA_DIR / "stackoverflow_electronics_main.txt",
                             sep='\t', header=None, names=['source', 'target'])
        
        return user_df, building_profiles, importance_df, user_assignments, summary, edges_df, None
    
    except Exception as e:
        return None, None, None, None, None, None, [str(e)]

@st.cache_resource
def load_model():
    """Load ML model"""
    try:
        with open(MODELS_DIR / "building_predictor.pkl", 'rb') as f:
            model_data = pickle.load(f)
        return model_data['vectorizer'], model_data['classifier']
    except:
        return None, None

def show_setup_section(missing_files):
    """Show setup section with run buttons"""
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Setup Required</h3>
        <p>Some data files are missing. Click the buttons below to generate them automatically.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if missing_files:
        st.write("**Missing files:**")
        for f in missing_files:
            st.write(f"• {f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Step 1: K-Core Analysis")
        if st.button("▶️ Run K-Core Analysis", key="run_kcore"):
            if run_script("QuickPrediction.py", "K-Core Analysis"):
                st.rerun()
        
        st.markdown("### 🏷️ Step 2: Tag Extraction")
        if st.button("▶️ Run Tag Extraction", key="run_tags"):
            if run_script("BuildingTagExtractor.py", "Tag Extraction"):
                st.rerun()
    
    with col2:
        st.markdown("### 🤖 Step 3: Train Predictor")
        if st.button("▶️ Train ML Model", key="run_predictor"):
            if run_script("BuildingPredictor.py", "ML Model Training"):
                st.rerun()
        
        st.markdown("### 🏆 Step 4: Building Importance")
        if st.button("▶️ Analyze Building Importance", key="run_importance"):
            if run_script("BuildingImportance.py", "Building Importance Analysis"):
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
    <div class="success-box">
        <p><strong>💡 Tip:</strong> Run the scripts in order (Step 1 → Step 2 → Step 3 → Step 4). 
        Each step takes a few minutes. The page will reload automatically when complete.</p>
    </div>
    """, unsafe_allow_html=True)

def test_model_accuracy():
    """Test model and display results"""
    st.markdown('<h2 style="color: #58a6ff;">🤖 Model Accuracy Testing</h2>', unsafe_allow_html=True)
    
    with st.spinner("Testing model accuracy..."):
        vectorizer, classifier = load_model()
        
        if vectorizer is None:
            st.error("Model not found. Please run Step 3 first.")
            return
        
        user_df, building_profiles, _, user_assignments, _, _, _ = load_all_data()
        
        # Create test samples
        X_test = []
        y_test = []
        
        sample_size = min(1000, len(user_assignments))
        user_sample = user_assignments.sample(n=sample_size, random_state=42)
        
        for _, user in user_sample.iterrows():
            building = user['building']
            if building in building_profiles:
                tags = building_profiles[building].get('top_10_tags', [])
                if tags:
                    tag_string = ' '.join(tags[:5])
                    X_test.append(tag_string)
                    y_test.append(building)
        
        # Vectorize and predict
        X_test_vec = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_vec)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
        
        with col2:
            st.metric("Test Samples", len(X_test))
        
        with col3:
            st.metric("Classes", len(classifier.classes_))
        
        # Classification report
        st.markdown("### 📊 Detailed Performance Metrics")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df[report_df.index.str.startswith('wavemap')].head(10)
        report_df['building'] = report_df.index
        report_df = report_df[['building', 'precision', 'recall', 'f1-score', 'support']]
        report_df.columns = ['Building', 'Precision', 'Recall', 'F1-Score', 'Support']
        st.dataframe(report_df, use_container_width=True)
        
        # Confusion matrix
        st.markdown("### 🎯 Confusion Matrix (Top 10 Buildings)")
        unique_buildings = sorted(set(y_test) | set(y_pred))
        top_buildings = unique_buildings[:10]
        
        mask = [y in top_buildings for y in y_test]
        y_test_top = [y for y, m in zip(y_test, mask) if m]
        y_pred_top = [y for y, m in zip(y_pred, mask) if m]
        
        if len(y_test_top) > 0:
            cm = confusion_matrix(y_test_top, y_pred_top, labels=top_buildings)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=[b.replace('wavemap_', '') for b in top_buildings],
                y=[b.replace('wavemap_', '') for b in top_buildings],
                colorscale='Viridis',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Building",
                yaxis_title="True Building",
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9')
            )
            
            st.plotly_chart(fig, use_container_width=True)

def analyze_building_importance():
    """Analyze and display building importance"""
    st.markdown('<h2 style="color: #58a6ff;">🏆 Building Importance Analysis (PageRank)</h2>', unsafe_allow_html=True)
    
    with st.spinner("Computing building importance..."):
        user_df, building_profiles, importance_df, user_assignments, _, edges_df, _ = load_all_data()
        
        # Display top buildings
        st.markdown("### 📊 Top 10 Most Important Buildings")
        
        top_10 = importance_df.head(10)
        
        # Create bar chart
        fig = go.Figure(go.Bar(
            x=top_10['pagerank'],
            y=top_10['building'],
            orientation='h',
            marker=dict(
                color=top_10['pagerank'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="PageRank")
            ),
            text=[f"{x:.4f}" for x in top_10['pagerank']],
            textposition='auto'
        ))
        
        fig.update_layout(
            xaxis_title="PageRank Score",
            yaxis_title="Building",
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.markdown("### 📋 Complete Rankings")
        display_df = importance_df[['rank', 'building', 'pagerank', 'betweenness', 'total_degree']].copy()
        display_df.columns = ['Rank', 'Building', 'PageRank', 'Betweenness', 'Total Connections']
        st.dataframe(display_df, use_container_width=True)

# Callbacks for state management
def toggle_accuracy():
    st.session_state.show_accuracy = not st.session_state.show_accuracy

def toggle_importance():
    st.session_state.show_importance = not st.session_state.show_importance

def main():
    """Main dashboard"""
    
    # Initialize Session State
    if 'show_accuracy' not in st.session_state:
        st.session_state.show_accuracy = False
    if 'show_importance' not in st.session_state:
        st.session_state.show_importance = False

    # Header
    st.markdown('<div class="main-header">🏙️ Stack Overflow Electronics Graph City</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Stack Overflow Electronics Knowledge-Verse • 15 Years of Community Data (2009-2024)</div>', unsafe_allow_html=True)
    
    # Load data
    user_df, building_profiles, importance_df, user_assignments, summary, edges_df, errors = load_all_data()
    
    if user_df is None or errors:
        show_setup_section(errors)
        return
    
    # === KEY METRICS ===
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Active Users", f"{len(user_df):,}")
    
    with col2:
        st.metric("🔗 Interactions", f"{len(edges_df):,}")
    
    with col3:
        st.metric("🏢 Buildings", len(building_profiles))
    
    with col4:
        st.metric("🏆 Max K-Core", summary['max_k_core'])
    
    with col5:
        st.metric("🎯 Clustering", "0.625")
    
    st.markdown("---")
    
    # === ACTION BUTTONS ===
    st.markdown('<h2 style="color: #58a6ff;">⚡ Quick Actions</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Use callbacks to toggle state instead of rendering directly in 'if st.button'
    with col1:
        btn_label_acc = "❌ Hide Model Accuracy" if st.session_state.show_accuracy else "🤖 Test Model Accuracy"
        st.button(btn_label_acc, use_container_width=True, on_click=toggle_accuracy)
    
    with col2:
        btn_label_imp = "❌ Hide Building Importance" if st.session_state.show_importance else "🏆 Analyze Building Importance"
        st.button(btn_label_imp, use_container_width=True, on_click=toggle_importance)
    
    st.markdown("---")

    # === CONDITIONALLY RENDER SECTIONS IN TWO COLUMNS ===
    # Create two columns for the graphs
    graph_col1, graph_col2 = st.columns(2)

    # Place content in the left column
    with graph_col1:
        if st.session_state.show_accuracy:
            test_model_accuracy()
            
    # Place content in the right column
    with graph_col2:
        if st.session_state.show_importance:
            analyze_building_importance()
            
    # Add a separator if either graph is visible
    if st.session_state.show_accuracy or st.session_state.show_importance:
        st.markdown("---")
    
    # === NETWORK ANALYSIS ===
    st.markdown('<h2 style="color: #58a6ff;">📊 Network Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### K-Core Distribution")
        k_core_counts = user_df['k_core'].value_counts().sort_index(ascending=False).head(15)
        
        fig = go.Figure(go.Bar(
            x=k_core_counts.index,
            y=k_core_counts.values,
            marker_color='#58a6ff'
        ))
        fig.update_layout(
            xaxis_title="K-Core Value",
            yaxis_title="Number of Users",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Expertise Level Distribution")
        expertise_counts = user_df['expertise_level'].value_counts()
        
        fig = go.Figure(go.Pie(
            labels=expertise_counts.index,
            values=expertise_counts.values,
            hole=0.4
        ))
        fig.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === TOP USERS ===
    st.markdown('<h2 style="color: #58a6ff;">🏆 Top Elite Users</h2>', unsafe_allow_html=True)
    
    top_users = user_df.head(20).copy()
    display_cols = ['user_id', 'k_core', 'degree', 'expertise_level']
    if 'original_user_id' in top_users.columns:
        display_cols.insert(1, 'original_user_id')
    if 'username' in top_users.columns:
        display_cols.insert(2, 'username')
    
    top_users_display = top_users[display_cols].copy()
    top_users_display.columns = [c.replace('_', ' ').title() for c in top_users_display.columns]
    
    st.dataframe(top_users_display, use_container_width=True, hide_index=True)
    
    # === INTERACTIVE PREDICTOR ===
    st.markdown('<h2 style="color: #58a6ff;">🔮 Building Predictor</h2>', unsafe_allow_html=True)
    
    vectorizer, classifier = load_model()
    
    if vectorizer is not None:
        st.markdown("### Predict Your Building")
        
        user_input = st.text_input(
            "Enter tags (space-separated):",
            placeholder="arduino sensors i2c programming"
        )
        
        if user_input:
            tags = user_input.strip().split()
            
            # Predict
            tag_string = ' '.join(tags)
            X_vec = vectorizer.transform([tag_string])
            prediction = classifier.predict(X_vec)[0]
            probabilities = classifier.predict_proba(X_vec)[0]
            
            buildings = classifier.classes_
            prob_dict = {building: float(prob) for building, prob in zip(buildings, probabilities)}
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            k_core = int(prediction.split('_')[1])
            profile = building_profiles.get(prediction, {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Building", prediction)
            
            with col2:
                st.metric("K-Core Level", k_core)
            
            with col3:
                st.metric("Confidence", f"{sorted_probs[0][1]:.1%}")
            
            st.write(f"**Primary Domain:** {profile.get('primary_domain', 'Unknown')}")
            st.write(f"**Building's Top Tags:** {', '.join(profile.get('top_5_tags', []))}")
            
            st.write("**Top 3 Predictions:**")
            for building, prob in sorted_probs[:3]:
                k = int(building.split('_')[1])
                st.write(f"• {building} (k-core={k}): {prob:.1%}")
    
    # === FOOTER ===
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #8b949e; padding: 2rem;'>
        <p><strong>Stack Overflow Electronics Graph City</strong></p>
        <p>CS 439 Semester Project • 2024</p>
        <p>Data: Stack Overflow Electronics (2009-2024) • 24,541 users • 146,084 interactions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()