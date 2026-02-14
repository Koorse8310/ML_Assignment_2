"""
Main script to train all classification models.
Execute this script to train and save all models.
"""

from src.model_training.classifier_trainer import CardiovascularClassifierTrainer

if __name__ == "__main__":
    # Initialize trainer with data paths
    model_trainer = CardiovascularClassifierTrainer(
        data_path="data/heart_disease_data.csv",
        output_dir="trained_models"
    )
    
    # Train all models
    model_trainer.train_all_models()
