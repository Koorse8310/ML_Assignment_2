# Setup Instructions

## Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Models
Run the training script to generate all model files:
```bash
python train_classifiers.py
```

This will create the following files in the `trained_models/` directory:
- `logistic_regression.pkl`
- `decision_tree.pkl`
- `k_nearest_neighbors.pkl`
- `gaussian_naive_bayes.pkl`
- `random_forest.pkl`
- `xgboost_classifier.pkl`
- `feature_normalizer.pkl`
- `model_performance.csv`

### Step 3: Run the Streamlit Application
```bash
streamlit run streamlit_app.py
```

The application will open automatically in your web browser.

## File Structure After Setup

```
ml-assignment-2-main/
├── streamlit_app.py
├── train_classifiers.py
├── data/
│   ├── heart_disease_data.csv
│   └── test_samples.csv
├── trained_models/          (created after training)
│   ├── *.pkl files
│   └── model_performance.csv
└── src/
    ├── model_training/
    └── web_interface/
```

## Important Notes

- Make sure to train the models before running the Streamlit app
- The training process may take a few minutes depending on your system
- All model files are saved in the `trained_models/` directory
