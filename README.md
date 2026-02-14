# Customer Churn Prediction using PyTorch & Machine Learning

## Overview

This project implements an **end-to-end customer churn prediction pipeline** using both traditional machine learning models and a deep learning model built with **PyTorch**.

The objective is to predict whether a telecom customer will **churn (leave the service)** based on their demographic, account, and service usage data. The project follows a **production-style machine learning workflow** with feature engineering, class imbalance handling, cross-validation, model comparison, and explainability.

---

## Key Features

* End-to-end ML pipeline
* Data cleaning and preprocessing
* Feature engineering
* Class imbalance handling (SMOTE)
* Hyperparameter tuning
* Cross-validation
* Machine learning and deep learning models
* Model comparison using multiple metrics
* ROC and Precision–Recall curves
* Feature importance analysis
* SHAP-based model interpretability
* Model saving and loading
* Streamlit-ready deployment interface

---

## Dataset

**Telco Customer Churn Dataset**

The dataset contains customer information such as:

* Demographics
* Account details
* Services subscribed
* Monthly charges
* Tenure
* Churn status (target variable)

**Target variable:**

* `Churn = 1` → Customer left
* `Churn = 0` → Customer stayed

---

## Tech Stack

* Python
* PyTorch
* Scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn
* SHAP
* imbalanced-learn (SMOTE)
* Streamlit (optional deployment)

---

## Project Pipeline

### 1. Data Preprocessing

* Removed irrelevant columns
* Handled missing values
* Encoded categorical variables
* Feature scaling

### 2. Feature Engineering

Created new business-relevant features:

* **AvgMonthlySpend** = TotalCharges / tenure
* **IsNewCustomer** (tenure < 6 months)
* **ServicesCount** (number of services subscribed)

---

### 3. Class Imbalance Handling

* Applied **SMOTE** to balance churn vs non-churn classes.

---

### 4. Model Training

Trained and compared three models:

1. Logistic Regression
2. Random Forest (with hyperparameter tuning)
3. Neural Network (PyTorch)

---

### 5. Deep Learning Model (PyTorch)

**Architecture:**

* Input layer
* Dense layer (64 units, ReLU)
* Dropout layer
* Dense layer (32 units, ReLU)
* Output layer (Sigmoid)

**Loss:** Binary Cross-Entropy
**Optimizer:** Adam

---

### 6. Model Evaluation

Metrics used:

* Accuracy
* ROC-AUC score
* Precision–Recall AUC
* Confusion matrix
* Classification report

---

### 7. Cross-Validation

* Applied **Stratified K-Fold cross-validation**
* Ensured stable and reliable performance estimates

---

### 8. Model Interpretability

Used:

* Random Forest feature importance
* **SHAP** for global feature impact visualization

---

### 9. Model Comparison

| Model                    | Accuracy | ROC-AUC |
| ------------------------ | -------- | ------- |
| Logistic Regression      | 0.73     | 0.74    |
| Random Forest            | 0.77     | 0.71    |
| Neural Network (PyTorch) | 0.75     | 0.74    |

*(Replace with your actual results)*

---

### 10. Business Insights

Key findings from the analysis:

* Customers with short tenure have the highest churn risk.
* High monthly charges correlate with higher churn probability.
* Month-to-month contract customers churn more frequently.
* Customers with fewer subscribed services are more likely to leave.
* New customers (tenure < 6 months) represent the highest-risk segment.

**Recommendations:**

* Offer retention incentives to new customers.
* Encourage long-term contracts.
* Provide service bundles for high-risk users.

---

## Project Structure

```
customer-churn-project/
│
├── churn_project.ipynb
├── data/
│   └── telco_churn.csv
├── models/
│   └── churn_model.pth
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### Option 1: Google Colab

1. Open the notebook.
2. Upload the dataset.
3. Run all cells.

---

### Option 2: Run Locally

#### 1. Clone the repository

```
git clone https://github.com/yourusername/customer-churn-project.git
cd customer-churn-project
```

#### 2. Install dependencies

```
pip install -r requirements.txt
```

#### 3. Run the notebook

```
jupyter notebook churn_project.ipynb
```

---

### Optional: Run Streamlit App

```
streamlit run app.py
```

---

## Future Improvements

* Neural network cross-validation
* Ensemble stacking
* Automated ML pipeline
* REST API deployment
* Docker containerization
