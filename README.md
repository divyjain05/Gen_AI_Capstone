# Intelligent Vehicle Maintenance Prediction

## Project Overview

The Intelligent Vehicle Maintenance Prediction project is a Machine Learning–based classification system designed to predict whether a vehicle requires maintenance based on structured maintenance and operational data.

The objective of this project is to build a complete end-to-end machine learning pipeline, including data preprocessing, feature engineering, model training, evaluation, and performance comparison using traditional machine learning algorithms.

---

## Project Structure

```
/Vehicle_Maintenance_Prediction/
│
├──── Vehicle_Maintenance_records.csv        # Dataset
│
├── app.py                                   # Deployment application
│
├── genai_capstone.py                        # Python implementation script
├── requirements.txt                         # Project dependencies
└── README.md                                # Project documentation
```

---

## Dataset

- **File Name:** `Vehicle_Maintenance_records.csv`
- **Type:** Structured tabular dataset
- **Target Variable:** Maintenance Required (Binary Classification)

The dataset contains vehicle-related attributes such as mileage, vehicle age, reported issues, engine size, odometer reading, insurance premium, service history, accident history, and fuel efficiency.

---

## Methodology

### 1. Data Loading
- Dataset loaded using Pandas.
- Initial inspection using `.head()`, `.info()`, and `.describe()`.

### 2. Data Cleaning
- Missing numerical values handled using median imputation.
- Data consistency verified across all columns.

### 3. Feature Engineering
- Selected key predictive features:
  - Mileage
  - Reported Issues
  - Vehicle Model
  - Engine Size
- One-Hot Encoding applied to categorical variable:
  - Vehicle Model
- Feature scaling applied using `StandardScaler`.

### 4. Train-Test Split
- 80% Training Data
- 20% Testing Data
- `random_state = 42` for reproducibility.

---

## Machine Learning Models

### Logistic Regression
- Solver: `liblinear`
- Penalty: `l2`

### Decision Tree Classifier
- Default parameters
- `random_state = 42`

---

## Model Evaluation

The following evaluation metrics were used:

- Accuracy Score
- Classification Report
- Confusion Matrix
- F1-Score (Class 1)

### Performance Summary

| Model                | Accuracy     | F1-Score (Class 1) |
|----------------------|--------------|--------------------|
| Logistic Regression  | Higher       | 0.89               |
| Decision Tree        | Competitive  | 0.84               |

Logistic Regression demonstrated better generalization performance based on evaluation metrics.

---

## Installation and Setup

### Option 1: Google Colab

1. Open Google Colab.
2. Upload `GenAI_Capstone.ipynb`.
3. Upload `Vehicle_Maintenance_records.csv` to the Colab environment.
4. Execute all cells sequentially.

### Option 2: Local Environment

#### Step 1: Clone the Repository

```bash
git clone https://github.com/divyjain05/Gen_AI_Capstone.git
cd Vehicle_Maintenance_Prediction
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate environment:

- Windows:
```bash
venv\\Scripts\\activate
```

- macOS/Linux:
```bash
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Run the Script

```bash
python genai_capstone.py
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Key Learnings

- Data preprocessing and missing value handling
- Feature encoding and scaling techniques
- Model training and evaluation
- Comparative analysis of classification algorithms
- Performance visualization

---

## Future Enhancements

- Hyperparameter tuning using GridSearchCV
- Cross-validation implementation
- Feature importance analysis
- Model serialization using Pickle or Joblib
- Deployment using Streamlit
- Integration of advanced ensemble models (Random Forest, XGBoost)

---

## Authors

- Divy Kumar Jain
- Utkarsh Jain
- Praveen Kumar Nitharwal
