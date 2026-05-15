# 📉 Telco Customer Churn Prediction

A production-aware machine learning pipeline that predicts customer churn using the IBM Telco Customer Churn dataset. Built with a focus on correct evaluation methodology, business interpretability, and model comparison across four model classes.

---

## 📌 Problem Statement

Customer churn is one of the most costly problems in the telecom industry. This project builds and compares multiple ML models to identify customers likely to churn, enabling targeted retention campaigns before the customer leaves.

**Key challenge:** The dataset is imbalanced (~73% No Churn / ~27% Churn), meaning naive models will simply predict "No Churn" for everyone and still appear accurate. This project handles that correctly.

---

## 📂 Dataset

- **Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 customers, 21 features
- **Target:** `Churn` — binary (Yes / No)
- **Features include:** tenure, contract type, internet service, monthly charges, total charges, payment method, and 15+ service subscription flags

---

## 🏗️ Project Pipeline

```
Raw Data
   │
   ├── Data Cleaning (TotalCharges coercion, customerID drop)
   ├── Label Encoding (categorical features)
   ├── Feature Engineering
   │       ├── AvgMonthlySpend = TotalCharges / (tenure + 1)
   │       ├── IsNewCustomer = (tenure < 6)
   │       └── ServicesCount = sum of all subscribed services
   │
   ├── Train / Test Split (80/20, stratified)
   ├── StandardScaler
   │
   ├── SMOTE (inside pipeline — no data leakage)
   │
   ├── Model Training + GridSearchCV
   │       ├── Logistic Regression
   │       ├── Random Forest
   │       ├── XGBoost
   │       └── Neural Network (PyTorch)
   │
   ├── Evaluation
   │       ├── ROC-AUC, Accuracy, F1, Precision, Recall
   │       ├── ROC Curve + Precision-Recall Curve
   │       ├── SHAP Feature Importance
   │       ├── Confusion Matrix
   │       └── Threshold Tuning
   │
   └── Business Summary
```

---

## ⚙️ Technical Highlights

### ✅ Correct SMOTE Implementation
SMOTE is wrapped inside an `imblearn.Pipeline` with `StratifiedKFold` cross-validation. This ensures synthetic samples are **never seen by validation folds**, preventing the data leakage that suppressed baseline scores by up to 0.13 AUC.

```python
rf_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42))
])
```

### ✅ XGBoost with scale_pos_weight
XGBoost handles class imbalance natively via `scale_pos_weight` (ratio of negative to positive class), which reweights the loss function directly rather than duplicating data.

### ✅ PyTorch Neural Network with Early Stopping
The NN uses BatchNorm, Dropout, a learning rate scheduler (`ReduceLROnPlateau`), and early stopping with patience=10 to prevent overfitting.

### ✅ Probability-based AUC
All ROC-AUC scores are computed using predicted **probabilities**, not binary predictions — a common mistake that artificially deflates AUC scores.

### ✅ Threshold Tuning
Default 0.5 threshold is not optimal for imbalanced churn data. The optimal threshold per model is found by maximizing F1 score across the precision-recall curve.

---

## 📊 Results

### Model Comparison

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.7395 | **0.8450** |
| Random Forest | 0.7466 | 0.8416 |
| XGBoost | 0.7410 | 0.8440 |
| Neural Network | 0.7537 | 0.8310 |

> **Best model by AUC:** Logistic Regression (0.845) — indicating that after feature engineering, the problem is largely linearly separable. This is a meaningful finding: model complexity did not outperform a well-tuned linear baseline.

---

### Classification Report — Best Model (Logistic Regression)

```
              precision    recall  f1-score   support

    No Churn       0.90      0.73      0.80      1035
       Churn       0.51      0.78      0.61       374

    accuracy                           0.74      1409
   macro avg       0.70      0.75      0.71      1409
weighted avg       0.80      0.74      0.75      1409
```

---

### Threshold Tuning Results

| Model | Best Threshold | Default F1 | Tuned F1 | Default Recall | Tuned Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.59 | 0.612 | **0.629** | 0.775 | 0.722 |
| Random Forest | 0.53 | 0.621 | **0.634** | 0.783 | 0.751 |
| XGBoost | 0.54 | 0.618 | **0.636** | 0.789 | 0.770 |

> All tuned thresholds are above 0.5, indicating models were slightly overconfident in churn predictions at the default cutoff. Tuning improved F1 by ~0.013–0.018 across all models.

---

## 💼 Business Interpretation

At the tuned threshold (LR, threshold = 0.59):

- **72% of actual churners** are correctly identified and flagged for retention
- When a customer is flagged as churning, the prediction is correct **~56% of the time**
- For every 100 churn alerts sent to a retention team:
  - ~56 are real churners that can be saved
  - ~44 are false alarms (wasted retention spend)

### Threshold Strategy

| Scenario | Recommended Threshold | Reasoning |
|---|---|---|
| Low-cost retention (discount email) | 0.40 | Maximize recall, catch ~85%+ of churners |
| High-cost retention (account manager call) | 0.59 | Maximize precision, reduce wasted effort |

---

## 🔍 Feature Importance (SHAP)

SHAP TreeExplainer was used on the Random Forest to explain individual predictions. Top features driving churn predictions:

- **tenure** — shorter tenure = higher churn risk
- **Contract type** — month-to-month contracts churn significantly more
- **MonthlyCharges** — higher charges correlate with churn
- **AvgMonthlySpend** (engineered) — captures value-per-month signal
- **TechSupport / OnlineSecurity** — absence of these services increases churn risk

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data | pandas, numpy |
| ML | scikit-learn, imbalanced-learn, XGBoost |
| Deep Learning | PyTorch |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn |
| Environment | Google Colab |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/telco-churn-prediction
cd telco-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook churn_prediction.ipynb
```

### requirements.txt
```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
torch
shap
matplotlib
seaborn
```

---

## 📁 Repository Structure

```
telco-churn-prediction/
│
├── churn_prediction.ipynb   # Main notebook
├── README.md
├── requirements.txt
├── churn_model.pth          # Saved PyTorch model weights
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 🧠 Key Learnings

1. **SMOTE leakage is a real problem.** Applying SMOTE before cross-validation inflated training performance and suppressed test AUC by up to 0.13. Wrapping it in a pipeline fixed this.
2. **Simpler models can win on small tabular data.** Logistic Regression matched or outperformed XGBoost and Random Forest on this ~7k row dataset by AUC.
3. **Threshold tuning matters more than model selection** on imbalanced datasets. The gap between models (0.831–0.845 AUC) is smaller than the gain from proper threshold selection.
4. **Probability calibration is critical.** Computing AUC from binary predictions instead of probabilities was causing artificially low scores across the board.

---

## 👤 Author

**Rahul Mahour**
B.Tech — Artificial Intelligence & Data Science
University School of Automation and Robotics, Delhi

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourusername)
