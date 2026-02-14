"""
Machine Learning Classifier Training Module
This module handles the training and evaluation of multiple classification algorithms
for cardiovascular disease prediction.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class CardiovascularClassifierTrainer:
    """Main class for training and managing multiple classification models."""
    
    def __init__(self, data_path, output_dir="trained_models"):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the training dataset
            output_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.feature_scaler = None
        self.classifier_models = {}
        self.performance_metrics = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load dataset and prepare features and labels."""
        raw_data = pd.read_csv(self.data_path)
        
        # Separate features and target
        feature_columns = raw_data.drop("target", axis=1)
        label_column = raw_data["target"]
        
        return feature_columns, label_column
    
    def split_dataset(self, features, labels, test_proportion=0.2, seed=42):
        """Split data into training and testing sets."""
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=test_proportion, random_state=seed, stratify=labels
        )
        return train_features, test_features, train_labels, test_labels
    
    def normalize_features(self, train_data, test_data=None):
        """Apply feature scaling using StandardScaler."""
        self.feature_scaler = StandardScaler()
        normalized_train = self.feature_scaler.fit_transform(train_data)
        
        if test_data is not None:
            normalized_test = self.feature_scaler.transform(test_data)
            return normalized_train, normalized_test
        return normalized_train
    
    def initialize_classifiers(self):
        """Initialize all classification models with their configurations."""
        self.classifier_models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            "decision_tree": DecisionTreeClassifier(
                random_state=42,
                max_depth=10
            ),
            "k_nearest_neighbors": KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            "gaussian_naive_bayes": GaussianNB(),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ),
            "xgboost_classifier": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_estimators=100
            )
        }
    
    def train_single_model(self, model_name, model_instance, train_X, train_y, test_X, test_y):
        """Train a single model and evaluate its performance."""
        # Train the model
        model_instance.fit(train_X, train_y)
        
        # Make predictions
        predicted_labels = model_instance.predict(test_X)
        
        # Get prediction probabilities if available
        if hasattr(model_instance, "predict_proba"):
            prediction_probs = model_instance.predict_proba(test_X)[:, 1]
        else:
            prediction_probs = predicted_labels.astype(float)
        
        # Calculate metrics
        model_metrics = {
            "Classifier": model_name,
            "Accuracy": round(accuracy_score(test_y, predicted_labels), 3),
            "AUC_ROC": round(roc_auc_score(test_y, prediction_probs), 3),
            "Precision": round(precision_score(test_y, predicted_labels), 3),
            "Recall": round(recall_score(test_y, predicted_labels), 3),
            "F1_Score": round(f1_score(test_y, predicted_labels), 3),
            "Matthews_Correlation": round(matthews_corrcoef(test_y, predicted_labels), 3)
        }
        
        # Save the trained model
        model_filepath = os.path.join(self.output_dir, f"{model_name}.pkl")
        joblib.dump(model_instance, model_filepath)
        
        return model_metrics
    
    def train_all_models(self):
        """Train all classifiers and save results."""
        # Load and prepare data
        feature_data, label_data = self.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_dataset(
            feature_data, label_data
        )
        
        # Normalize features
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        # Save the scaler
        scaler_path = os.path.join(self.output_dir, "feature_normalizer.pkl")
        joblib.dump(self.feature_scaler, scaler_path)
        
        # Initialize models
        self.initialize_classifiers()
        
        # Train each model
        for model_name, model_obj in self.classifier_models.items():
            print(f"Training {model_name}...")
            metrics = self.train_single_model(
                model_name, model_obj, X_train_scaled, y_train, 
                X_test_scaled, y_test
            )
            self.performance_metrics.append(metrics)
        
        # Save performance results
        results_dataframe = pd.DataFrame(self.performance_metrics)
        results_path = os.path.join(self.output_dir, "model_performance.csv")
        results_dataframe.to_csv(results_path, index=False)
        
        print("\n[SUCCESS] All models trained successfully!")
        print("\nPerformance Summary:")
        print(results_dataframe.to_string(index=False))
        
        return results_dataframe


def main():
    """Main execution function."""
    # Configuration
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_file = os.path.join(base_dir, "data", "heart_disease_data.csv")
    models_directory = os.path.join(base_dir, "trained_models")
    
    # Create trainer instance
    trainer = CardiovascularClassifierTrainer(
        data_path=dataset_file,
        output_dir=models_directory
    )
    
    # Train all models
    trainer.train_all_models()


if __name__ == "__main__":
    main()
