# Vehicle Maintenance Risk Prediction System

The Intelligent Vehicle Maintenance Prediction project is a Machine Learning–based classification system designed to predict whether a vehicle requires maintenance based on structured maintenance and operational data.

---

## Objective

Design and implement a supervised ML system that:

- Accepts historical vehicle maintenance data  
- Performs preprocessing and feature engineering  
- Handles class imbalance using SMOTE  
- Predicts maintenance requirement probability  
- Provides a working Streamlit UI for real-time inference  

---

## Dataset

**Source:** Kaggle – Vehicle Maintenance Records Dataset  
**Target Variable:** `Need_Maintenance` (Binary Classification)

- 0 → No maintenance required  
- 1 → Maintenance required  

---

## Dataset Modifications

To simulate realistic operational conditions:

- Date columns were removed as they were not directly useful for modeling.
- Missing values were imputed using median (numeric) and mode (categorical).
- Extreme outliers (top & bottom 10% in selected features) were removed to reduce deterministic separation.
- Feature engineering was applied (`mileage_per_year`) to improve predictive signal.
- One-hot encoding was performed for categorical variables.

Class imbalance (~80% maintenance cases) was addressed using:

**SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE was applied only on training data to avoid data leakage.

---

## Technical Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Imbalanced-learn (SMOTE)  
- Streamlit  

---

## Models Evaluated

- Logistic Regression (Regularized)  
- Decision Tree (Depth-limited)  

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  

---

## Model Selection Strategy

In predictive maintenance systems, false negatives are costly.

Failing to predict a maintenance requirement may lead to:

- Unexpected breakdowns  
- Safety risks  
- Higher repair costs  
- Operational downtime  

Therefore, **Recall** was prioritized during evaluation.

A custom probability threshold of **0.7** was selected to:

- Reduce false positives  
- Maintain strong recall  
- Provide high-confidence maintenance alerts  

---

## Model Performance (Post-Tuning)

### Logistic Regression
- ROC-AUC ≈ 0.90–0.95  
- Balanced recall with regularization (C=0.3)  

### Decision Tree
- ROC-AUC ≈ 0.85–0.92  
- Depth-limited to prevent overfitting  
- Improved interpretability  

Although Logistic Regression produced slightly stronger ROC-AUC scores, both models demonstrated competitive performance.

---

## Final Deployment Models

Both Logistic Regression and Decision Tree are available in the deployed application.

### Risk Categorization

- Probability < 0.4 → LOW  
- 0.4 – 0.7 → MODERATE  
- > 0.7 → HIGH  

---

## Deployment Features

The deployed model uses:

- Mileage  
- Vehicle Age  
- Reported Issues  
- Engine Size  
- Service History  
- Odometer Reading  
- Derived Feature: `mileage_per_year`  

The application provides:

- Real-time probability prediction  
- Binary maintenance classification  
- Risk level categorization  
- Dataset-level risk analytics  

---

## Project Structure

```
.
├── app.py  
├── requirements.txt  
├── cleaned_vehicle_data.csv  
├── logistic_model_balanced.pkl  
├── decision_tree_balanced.pkl  
├── scaler.pkl  
├── feature_columns.pkl  
└── notebook/
    └── Vehicle_Maintenance_Model.ipynb  

```

---

## Hosted Link
https://genaicapstonedivy.streamlit.app

---

## Authors

- Divy Kumar Jain
- Utkarsh Jain
- Praveen Kumar Nitharwal