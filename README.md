# Telecom Customer Churn Prediction

A complete end-to-end machine learning project that predicts customer churn for a telecom company — from exploratory data analysis through model comparison, explainability, economics-driven business strategy, and A/B test design.

## 📋 Project Overview

Customer churn (attrition) is a critical business challenge in the telecom industry. This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 records, 21 features) to:

1. **Explore** customer demographics, service usage, and billing patterns
2. **Identify** key drivers of churn through EDA and feature importance analysis
3. **Build** three predictive models (Logistic Regression, Random Forest, XGBoost) with rigorous evaluation
4. **Validate** using a 3-way stratified split (Train 60% / Test 20% / Validation 20%) and 5-fold cross-validation
5. **Explain** model predictions using SHAP (SHapley Additive exPlanations) for global and individual-level interpretability
6. **Quantify** business impact through unit economics simulation, ROI sensitivity analysis, and profit-maximizing threshold selection
7. **Design** a retention A/B test with statistical power analysis and bootstrap simulation
8. **Recommend** actionable, dollar-denominated retention strategies

## 📊 Key Results

### Model Performance (Test Set)

| Metric | Logistic Regression | Random Forest | XGBoost | Winner |
|--------|:---:|:---:|:---:|:---:|
| Accuracy | 0.802 | 0.767 | 0.759 | LR |
| Precision (Churn) | 0.658 | 0.544 | 0.534 | LR |
| **Recall (Churn)** | 0.529 | **0.749** | 0.730 | **RF** |
| F1-Score | 0.587 | 0.630 | 0.617 | RF |
| AUC | 0.836 | 0.837 | 0.825 | RF |
| CV AUC (5-fold) | 0.849 ± 0.012 | 0.849 ± 0.013 | 0.832 ± 0.011 | LR / RF |

**Final Validation (Random Forest):** AUC = 0.844, Churn Recall = 75%, confirming strong generalization with no overfitting.

### Business Economics Highlights

| Metric | Value |
|--------|-------|
| Optimal threshold (τ) | 0.36 (F1-maximized) |
| Expected profit (validation set) | ~$53K across cost assumptions |
| Profitable retention cost range | $20 – $100 per customer |
| A/B test simulated churn reduction | 14.5% (95% CI: 7.3% – 21.8%) |
| Estimated annual revenue saved | ~$42,605 |
| Net ROI after campaign costs | ~$26,055/year |

## 🔍 Key Findings

- **Tenure** is the strongest predictor — new customers are most vulnerable
- **Month-to-month contracts** have dramatically higher churn rates
- **Higher monthly charges** correlate with increased attrition
- **Electronic check payment** is associated with higher churn
- **Lack of support services** (OnlineSecurity, TechSupport) increases churn risk
- **Fiber optic customers** churn at 2.2× the rate of DSL, suggesting a pricing or service quality gap
- The retention model is **profitable across a wide range of assumptions**, with clear go/no-go decision criteria from the A/B test framework

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data manipulation
- **NumPy** — numerical computing
- **Matplotlib / Seaborn** — data visualization
- **scikit-learn** — machine learning (Logistic Regression, Random Forest, cross-validation, metrics)
- **XGBoost** — gradient boosting classifier
- **SHAP** — model explainability (global & local explanations)
- **SciPy** — statistical power analysis and hypothesis testing

## 📁 Project Structure

```
churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
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
| 7. Business Economics & Unit Economics | CLV/ROI sensitivity, profit-maximizing threshold, SHAP→business strategy, segment analysis |
| 8. A/B Test Design & Simulation | Power analysis, bootstrap retention experiment, go/no-go decision framework |

## 📝 License

This project is for educational and portfolio purposes.
