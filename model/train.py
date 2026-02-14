"""
Training script for all classification models
This script trains all 6 classification models and saves them to the model directory.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path to access dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def train_all_models():
    """Train all classification models and save them."""
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.csv")
    data = pd.read_csv(dataset_path)
    
    # Separate features and target
    X = data.drop("target", axis=1)
    y = data["target"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    model_dir = os.path.dirname(__file__)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("Scaler saved successfully.")
    
    # Define all models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "knn": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=100)
    }
    
    # Store results
    results = []
    
    # Train each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Get probabilities for AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred.astype(float)
        
        # Calculate metrics
        metrics = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "AUC": round(roc_auc_score(y_test, y_prob), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1": round(f1_score(y_test, y_pred), 3),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 3)
        }
        results.append(metrics)
        
        # Save model
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"  {name} saved successfully.")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(model_dir, "model_performance.csv")
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    print(f"\nAll models saved in: {model_dir}")
    
    return results_df


if __name__ == "__main__":
    train_all_models()
