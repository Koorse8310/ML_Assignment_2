# Machine Learning Classification Models – Heart Disease Prediction

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease based on patient medical attributes. The problem involves binary classification where the target variable indicates whether a patient has heart disease (1) or not (0). This is a critical healthcare application where accurate prediction can aid in early diagnosis and treatment planning. The project implements six different classification algorithms and evaluates their performance using various metrics to identify the most suitable model for this prediction task.

## b. Dataset Description

The dataset used in this project is a publicly available heart disease dataset containing medical and clinical attributes of patients. The dataset consists of **1,025 instances** with **13 feature columns** and one target variable. 

**Features included:**
- **age**: Age of the patient in years
- **sex**: Gender (0 = female, 1 = male)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol level (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (0 = no, 1 = yes)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia type (0-3)

**Target Variable:**
- **target**: Binary classification label (0 = No heart disease, 1 = Heart disease present)

The dataset was split into training (80%) and testing (20%) sets using stratified sampling to maintain class distribution. Feature scaling was applied using StandardScaler to normalize all features for optimal model performance.

## c. Models Used

The following six classification models were implemented and evaluated on the same dataset:

### Comparison Table with Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.810 | 0.930 | 0.762 | 0.914 | 0.831 | 0.631 |
| Decision Tree | 0.985 | 0.986 | 1.000 | 0.971 | 0.986 | 0.971 |
| kNN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Naive Bayes | 0.829 | 0.904 | 0.807 | 0.876 | 0.840 | 0.660 |
| Random Forest (Ensemble) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost (Ensemble) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieved moderate performance with an accuracy of 81.0% and AUC of 0.930. The model shows good recall (0.914) indicating it can identify most positive cases, but lower precision (0.762) suggests some false positives. This linear model works well for capturing linear relationships in the data but may struggle with complex non-linear patterns. The F1-score of 0.831 and MCC of 0.631 indicate balanced but not exceptional performance. |
| Decision Tree | Decision Tree demonstrated excellent performance with 98.5% accuracy and near-perfect metrics (AUC: 0.986, Precision: 1.000, Recall: 0.971, F1: 0.986, MCC: 0.971). The model effectively captures decision boundaries through recursive partitioning. However, the perfect precision with slightly lower recall suggests the model is conservative in positive predictions, which is desirable for medical applications. The high MCC value (0.971) indicates strong correlation between predicted and actual classes. |
| kNN | K-Nearest Neighbors achieved perfect performance metrics (100% across all metrics) on the test set. This exceptional performance suggests the model successfully leverages local patterns in the feature space. The distance-weighted approach with k=5 neighbors effectively captures the underlying data distribution. However, perfect scores may indicate potential overfitting to the test set, and the model's performance might vary with different data splits or in real-world scenarios. |
| Naive Bayes | Gaussian Naive Bayes showed solid performance with 82.9% accuracy and an AUC of 0.904. The model achieved balanced precision (0.807) and recall (0.876), resulting in an F1-score of 0.840. The assumption of feature independence, while not strictly true for medical data, still allows the model to perform reasonably well. The MCC of 0.660 indicates moderate correlation. This model is computationally efficient and provides probabilistic outputs, making it useful for applications requiring fast inference. |
| Random Forest (Ensemble) | Random Forest achieved perfect performance metrics (100% across all evaluation metrics) on the test dataset. As an ensemble method combining multiple decision trees, it effectively reduces overfitting while maintaining high predictive power. The model's ability to capture complex interactions between features through multiple tree voting contributes to its excellent performance. The perfect scores suggest the ensemble successfully generalizes the patterns in the training data to the test set, though cross-validation would be recommended to ensure robustness. |
| XGBoost (Ensemble) | XGBoost, a gradient boosting ensemble method, also achieved perfect performance metrics (100% across all metrics). The sequential ensemble approach, where each tree corrects errors from previous trees, enables the model to learn complex patterns effectively. XGBoost's regularization techniques and optimized implementation contribute to its strong performance. Like Random Forest, the perfect scores indicate excellent generalization on the test set, with the added benefit of gradient-based optimization that can capture subtle patterns in the data. |

---

## Project Structure

```
ml-assignment-2-main/
│
├── streamlit_app.py              # Main Streamlit application (includes model training)
├── requirements.txt              # Python package dependencies
├── README.md                     # Project documentation
├── dataset.csv                   # Training dataset
├── test.csv                      # Sample test dataset (available for download)
│
└── model/                        # Trained model files
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    └── model_performance.csv
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit application:**
   ```bash
   streamlit run streamlit_app.py
   ```
   The models will be automatically trained on first run if they don't exist. Model files will be saved in the `model/` directory.

## Usage

### Running the Streamlit Application

Launch the web dashboard using:
```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser, typically at `http://localhost:8501`

### Using the Dashboard

1. **Download Test Dataset**: Use the download button to get the sample test dataset
2. **Select Model**: Choose from the dropdown menu to select a classification model
3. **Upload Data**: Upload your test CSV file (must include a 'target' column)
4. **View Results**: 
   - Performance metrics (Accuracy, Precision, Recall, F1-Score)
   - Detailed classification report
   - Confusion matrix visualization

## Deployment

This application can be deployed on Streamlit Community Cloud:

1. Push your code to a GitHub repository
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy the app using `streamlit_app.py` as the main file

## Notes

- Models are automatically trained on first run if they don't exist in the `model/` directory
- The test dataset must contain the same feature columns as the training data
- The 'target' column is required in test datasets for evaluation
- You can download the sample test dataset (`test.csv`) directly from the Streamlit app

## License

This project is created for educational purposes as part of a machine learning assignment.
