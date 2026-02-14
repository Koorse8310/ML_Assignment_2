"""
Streamlit Application for Cardiovascular Disease Prediction
This application implements multiple classification models and provides
an interactive interface for model evaluation.
"""

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import os

# Model training imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Configure page settings
st.set_page_config(
    page_title="Cardiovascular Disease Prediction Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class ModelTrainer:
    """Class for training classification models."""
    
    def __init__(self, data_path, output_dir="model"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.scaler = None
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_models(self):
        """Train all classification models."""
        # Load data
        data = pd.read_csv(self.data_path)
        X = data.drop("target", axis=1)
        y = data["target"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.output_dir, "scaler.pkl"))
        
        # Define models
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=10),
            "knn": KNeighborsClassifier(n_neighbors=5, weights='distance'),
            "naive_bayes": GaussianNB(),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            "xgboost": XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=100)
        }
        
        # Train each model
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            joblib.dump(model, os.path.join(self.output_dir, f"{name}.pkl"))


def check_models_exist():
    """Check if all model files exist."""
    model_dir = Path("model")
    required_files = [
        "logistic_regression.pkl", "decision_tree.pkl", "knn.pkl",
        "naive_bayes.pkl", "random_forest.pkl", "xgboost.pkl", "scaler.pkl"
    ]
    return all((model_dir / f).exists() for f in required_files)


def train_models_if_needed():
    """Train models if they don't exist."""
    if not check_models_exist():
        with st.spinner("Training models for the first time. This may take a few minutes..."):
            try:
                trainer = ModelTrainer(data_path="dataset.csv", output_dir="model")
                trainer.train_models()
                st.success("Models trained successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
                st.stop()


def load_model(model_name):
    """Load a trained model and scaler."""
    train_models_if_needed()
    model_path = Path("model") / f"{model_name}.pkl"
    scaler_path = Path("model") / "scaler.pkl"
    
    if not model_path.exists():
        st.error(f"Model {model_name} not found.")
        st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def main():
    """Main application."""
    # Header
    st.markdown('<div class="main-header">🫀 Cardiovascular Disease Prediction System</div>', 
               unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    An interactive machine learning platform for predicting cardiovascular disease 
    using multiple classification algorithms.
    </div>
    """, unsafe_allow_html=True)
    
    # Download test dataset section
    st.markdown("---")
    st.subheader("📥 Download Sample Test Dataset")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Download the sample test dataset to evaluate the models:")
    with col2:
        try:
            with open("test.csv", "rb") as file:
                st.download_button(
                    label="⬇️ Download test.csv",
                    data=file,
                    file_name="test.csv",
                    mime="text/csv",
                    key="download_test"
                )
        except FileNotFoundError:
            st.info("Test dataset file not found.")
    
    # Model selection
    st.markdown("---")
    st.subheader("🔧 Model Configuration & Data Upload")
    
    model_options = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }
    
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_model = st.selectbox("Select Classification Model:", list(model_options.keys()))
    with col2:
        uploaded_file = st.file_uploader("Upload Test Dataset (CSV):", type=["csv"])
    
    # Process predictions
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            if "target" not in data.columns:
                st.error("Error: Dataset must contain a 'target' column.")
                return
            
            X = data.drop("target", axis=1)
            y = data["target"]
            
            # Load model
            model_name = model_options[selected_model]
            model, scaler = load_model(model_name)
            
            # Make predictions
            with st.spinner(f"Processing predictions using {selected_model}..."):
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
            
            st.success(f"Predictions completed using {selected_model}!")
            
            # Performance metrics
            st.markdown("---")
            st.subheader("📊 Model Performance Evaluation")
            
            report = classification_report(y, predictions, output_dict=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{report['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{report['1']['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{report['1']['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{report['1']['f1-score']:.3f}")
            
            with st.expander("📈 Detailed Classification Report"):
                st.json(report)
            
            # Confusion matrix
            st.markdown("---")
            st.subheader("🧮 Confusion Matrix Visualization")
            
            cm = confusion_matrix(y, predictions)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="YlOrRd",
                cbar_kws={'label': 'Count'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
            ax.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
            ax.set_ylabel("Actual Label", fontsize=12, fontweight='bold')
            ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=20)
            
            # Add class labels
            ax.set_xticklabels(['No Disease', 'Disease'])
            ax.set_yticklabels(['No Disease', 'Disease'])
            
            st.pyplot(fig)
            
            # Confusion matrix interpretation
            tn, fp, fn, tp = cm.ravel()
            st.info(f"""
            **Matrix Interpretation:**
            - True Negatives (TN): {tn} - Correctly predicted no disease
            - False Positives (FP): {fp} - Incorrectly predicted disease
            - False Negatives (FN): {fn} - Missed disease cases
            - True Positives (TP): {tp} - Correctly predicted disease
            """)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
