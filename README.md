# Telecom Customer Churn Prediction

A complete end-to-end machine learning project that predicts customer churn for a telecom company, from exploratory data analysis through model comparison, explainability, and business recommendations.

## 📋 Project Overview

Customer churn (attrition) is a critical business challenge in the telecom industry. This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 records, 21 features) to:

1. **Explore** customer demographics, service usage, and billing patterns
2. **Identify** key drivers of churn through EDA and feature importance analysis
3. **Build** three predictive models (Logistic Regression, Random Forest, XGBoost) with rigorous evaluation
4. **Validate** using a 3-way stratified split (Train 60% / Test 20% / Validation 20%) and 5-fold cross-validation
5. **Explain** model predictions using SHAP (SHapley Additive exPlanations) for global and individual-level interpretability
6. **Recommend** actionable retention strategies based on model insights

## 📊 Key Results

**Test Set Comparison:**

| Metric | Logistic Regression | Random Forest | XGBoost | Winner |
|--------|:---:|:---:|:---:|:---:|
| Accuracy | 0.802 | 0.767 | 0.759 | LR |
| Precision (Churn) | 0.658 | 0.544 | 0.534 | LR |
| **Recall (Churn)** | 0.529 | **0.749** | 0.730 | **RF** |
| F1-Score | 0.587 | 0.630 | 0.617 | RF |
| AUC | 0.836 | 0.837 | 0.825 | RF |
| CV AUC (5-fold) | 0.849 ± 0.012 | 0.849 ± 0.013 | 0.832 ± 0.011 | LR / RF |

**Final Validation (Random Forest):** AUC = 0.844, Churn Recall = 75%, confirming strong generalization with no overfitting.

**Random Forest is recommended** for deployment — it identifies 74.9% of actual churners, significantly outperforming the LR baseline (52.9%), with the highest F1-score and AUC.

## 🔍 Key Findings

- **Tenure** is the strongest predictor — new customers are most vulnerable
- **Month-to-month contracts** have dramatically higher churn rates
- **Higher monthly charges** correlate with increased attrition
- **Electronic check payment** is associated with higher churn
- **Lack of support services** (OnlineSecurity, TechSupport) increases churn risk

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data manipulation
- **NumPy** — numerical computing
- **Matplotlib / Seaborn** — data visualization
- **scikit-learn** — machine learning (Logistic Regression, Random Forest, cross-validation, metrics)
- **XGBoost** — gradient boosting classifier
- **SHAP** — model explainability (global & local explanations)

## 📁 Project Structure

```
churn-prediction/
├── README.md
├── requirements.txt
├── churn prediction.ipynb    # Main analysis notebook
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/churn-prediction.git
   cd churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` folder.

4. Open and run the notebook:
   ```bash
   jupyter notebook "churn prediction.ipynb"
   ```

## 📓 Notebook Structure

| Section | Description |
|---------|-------------|
| 1. Data Loading & Overview | Load dataset, inspect shape, types, and columns |
| 2. Data Cleaning | Handle missing values in TotalCharges |
| 3. Exploratory Data Analysis | Visualize churn patterns across demographics, services, contracts, and billing |
| 4. Data Preprocessing | Feature engineering, one-hot encoding, 3-way stratified split (60/20/20) |
| 5.1 Logistic Regression | Baseline model with 5-fold CV |
| 5.2 Random Forest | Ensemble model with class balancing and 5-fold CV |
| 5.3 XGBoost | Gradient boosting with scale_pos_weight and 5-fold CV |
| 5.4 SHAP Explainability | Global (summary plot) and local (waterfall) model explanations |
| 6. Model Comparison & Conclusion | 3-model comparison, final validation, business recommendations |

## 📝 License

This project is for educational and portfolio purposes.
