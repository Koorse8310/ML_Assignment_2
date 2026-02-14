"""
Interactive Web Dashboard for Cardiovascular Disease Prediction
This Streamlit application provides an intuitive interface for model evaluation
and prediction visualization.
"""

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class PredictionDashboard:
    """Main dashboard class for handling predictions and visualizations."""
    
    def __init__(self):
        # Paths relative to project root (where streamlit_app.py is located)
        self.models_base_path = Path("trained_models")
        self.scaler_path = self.models_base_path / "feature_normalizer.pkl"
        self.available_models = {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "K-Nearest Neighbors": "k_nearest_neighbors.pkl",
            "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
            "Random Forest": "random_forest.pkl",
            "XGBoost Classifier": "xgboost_classifier.pkl"
        }
    
    def load_model_and_scaler(self, model_filename):
        """Load the selected model and feature scaler."""
        model_path = self.models_base_path / model_filename
        scaler = joblib.load(self.scaler_path)
        model = joblib.load(model_path)
        return model, scaler
    
    def render_header(self):
        """Render the main header section."""
        st.markdown('<div class="main-header">🫀 Cardiovascular Disease Prediction System</div>', 
                   unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        An interactive machine learning platform for predicting cardiovascular disease 
        using multiple classification algorithms. Upload your test dataset and explore 
        model performance metrics and visualizations.
        </div>
        """, unsafe_allow_html=True)
    
    def render_dataset_download(self):
        """Render the test dataset download section."""
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Sample Test Dataset")
            st.write("Download the sample test dataset to evaluate the models:")
        
        with col2:
            try:
                with open("data/test_samples.csv", "rb") as file:
                    st.download_button(
                        label="⬇️ Download Test Dataset",
                        data=file,
                        file_name="test_samples.csv",
                        mime="text/csv",
                        key="download_test"
                    )
            except FileNotFoundError:
                st.info("Test dataset file not found. Please ensure test_samples.csv exists in the data folder.")
    
    def render_model_selection(self):
        """Render model selection and file upload interface."""
        st.markdown("---")
        st.subheader("🔧 Model Configuration & Data Upload")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_model_display = st.selectbox(
                "Select Classification Model:",
                options=list(self.available_models.keys()),
                help="Choose the machine learning model for prediction"
            )
        
        with col2:
            uploaded_dataset = st.file_uploader(
                "Upload Test Dataset (CSV format):",
                type=["csv"],
                help="Upload a CSV file containing test data with the target column"
            )
        
        return selected_model_display, uploaded_dataset
    
    def process_predictions(self, model, scaler, features, true_labels):
        """Process predictions and return results."""
        # Normalize features
        normalized_features = scaler.transform(features)
        
        # Generate predictions
        predicted_labels = model.predict(normalized_features)
        
        return predicted_labels
    
    def render_performance_metrics(self, true_labels, predicted_labels):
        """Render classification performance metrics."""
        st.markdown("---")
        st.subheader("📊 Model Performance Evaluation")
        
        # Generate classification report
        performance_report = classification_report(
            true_labels, predicted_labels, output_dict=True
        )
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{performance_report['accuracy']:.3f}")
        
        with col2:
            precision = performance_report['1']['precision']
            st.metric("Precision", f"{precision:.3f}")
        
        with col3:
            recall = performance_report['1']['recall']
            st.metric("Recall", f"{recall:.3f}")
        
        with col4:
            f1 = performance_report['1']['f1-score']
            st.metric("F1-Score", f"{f1:.3f}")
        
        # Detailed classification report
        with st.expander("📈 Detailed Classification Report"):
            st.json(performance_report)
    
    def render_confusion_matrix(self, true_labels, predicted_labels):
        """Render confusion matrix visualization."""
        st.markdown("---")
        st.subheader("🧮 Confusion Matrix Visualization")
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Create visualization
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
        
        # Interpretation
        tn, fp, fn, tp = cm.ravel()
        st.info(f"""
        **Matrix Interpretation:**
        - True Negatives (TN): {tn} - Correctly predicted no disease
        - False Positives (FP): {fp} - Incorrectly predicted disease
        - False Negatives (FN): {fn} - Missed disease cases
        - True Positives (TP): {tp} - Correctly predicted disease
        """)


def main():
    """Main application entry point."""
    dashboard = PredictionDashboard()
    
    # Render header
    dashboard.render_header()
    
    # Render dataset download
    dashboard.render_dataset_download()
    
    # Render model selection
    selected_model, uploaded_file = dashboard.render_model_selection()
    
    # Process if file is uploaded
    if uploaded_file is not None:
        try:
            # Load data
            test_data = pd.read_csv(uploaded_file)
            
            # Check for target column
            if "target" not in test_data.columns:
                st.error("❌ Error: The uploaded dataset must contain a 'target' column.")
                return
            
            # Separate features and labels
            feature_data = test_data.drop("target", axis=1)
            label_data = test_data["target"]
            
            # Load model and scaler
            model_filename = dashboard.available_models[selected_model]
            classifier_model, feature_scaler = dashboard.load_model_and_scaler(model_filename)
            
            # Show processing status
            with st.spinner(f"Processing predictions using {selected_model}..."):
                predictions = dashboard.process_predictions(
                    classifier_model, feature_scaler, feature_data, label_data
                )
            
            st.success(f"✅ Predictions completed using {selected_model}!")
            
            # Display results
            dashboard.render_performance_metrics(label_data, predictions)
            dashboard.render_confusion_matrix(label_data, predictions)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please ensure your dataset format matches the expected structure.")


if __name__ == "__main__":
    main()
