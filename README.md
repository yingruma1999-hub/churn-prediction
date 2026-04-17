<h1 align="center">📱 Telecom Customer Churn Prediction</h1>
<h3 align="center">From Predictive Modeling to Dollar-Denominated Business Strategy</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-Random%20Forest-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green" />
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet" />
  <img src="https://img.shields.io/badge/A%2FB%20Test-Power%20Analysis%20%7C%20Bootstrap-red" />
</p>

---

## 🎯 Executive Summary

> **The Problem:** A telecom company loses ~26% of its customers to churn. Identifying who will leave — and intervening profitably — is critical.
>
> **The Solution:** Built an end-to-end ML pipeline: **3 models compared → SHAP explainability → unit economics → profit-optimizing threshold → A/B test design**. The best model (Random Forest, AUC = 0.844) catches **75% of churners** and is profitable across retention costs from $20 to $100.
>
> **Business Impact:** Lowering the decision threshold from 0.50 → 0.36 increases expected profit by ~$53K. A simulated retention A/B test shows **14.5% churn reduction** with ~$26K net annual ROI.

---

## 📊 Key Results at a Glance

### Model Comparison (Test Set)

| Metric | Logistic Regression | Random Forest 🏆 | XGBoost |
|:-------|:-------------------:|:-----------------:|:-------:|
| **Churn Recall** | 52.9% | **74.9%** | 73.0% |
| AUC | 0.836 | **0.837** | 0.825 |
| CV AUC (5-fold) | 0.849 | **0.849** | 0.832 |
| Validation AUC | — | **0.844** | — |

<p align="center">
  <img src="assets/model_comparison.png" width="95%" alt="Model performance bar chart and ROC curves — Random Forest achieves highest AUC and recall" />
</p>

> **Random Forest** is the recommended model: highest churn recall (75%), best F1 (0.63), and validated AUC (0.844) confirming no overfitting.

---

## 🔍 What Drives Churn? (SHAP Explainability)

### Global Feature Impact

<p align="center">
  <img src="assets/shap_summary.png" width="80%" alt="SHAP summary plot showing tenure, contract type, and monthly charges as top churn drivers" />
</p>

> **Top 3 churn drivers:** Short tenure (new customers), month-to-month contracts, and high monthly charges. Customers without TechSupport or OnlineSecurity are also at elevated risk.

### Individual Prediction — Why This Customer Churned

<p align="center">
  <img src="assets/shap_waterfall.png" width="80%" alt="SHAP waterfall plot explaining a single customer's churn prediction" />
</p>

> SHAP waterfall plots make every prediction **transparent and actionable** — retention teams can see exactly which factors to address for each customer.

### Segment-Level Insight: Contract Type

<p align="center">
  <img src="assets/shap_contract_segment.png" width="85%" alt="SHAP impact comparison: month-to-month vs long-term contract customers" />
</p>

> Month-to-month customers show 3× higher SHAP impact from MonthlyCharges and tenure — they are **price-sensitive and flight-risk in early months**.

---

## 💰 Business Economics

### ROI Sensitivity — Profitable Across Assumptions

<p align="center">
  <img src="assets/roi_heatmap.png" width="85%" alt="ROI sensitivity heatmap: green (profitable) zone spans retention costs $20-$100" />
</p>

> The model is **profitable (green zone) across a wide range of operating conditions** — from $20 to $100 per retention contact, and across probability thresholds from 0.20 to 0.70.

### Profit-Maximizing Threshold

<p align="center">
  <img src="assets/threshold_optimization.png" width="85%" alt="Profit curve showing optimal threshold at τ=0.36, yielding ~$53K profit" />
</p>

> **Lowering the threshold from 0.50 → 0.36** captures more at-risk customers and **maximizes total expected profit at ~$53K** — a significant improvement over the default cutoff.

---

## 🧪 A/B Test Design & Simulation

### Simulated Retention Experiment (10,000 Bootstrap Iterations)

<p align="center">
  <img src="assets/ab_test_simulation.png" width="90%" alt="Bootstrap distribution of treatment effect and simulated churn rate distributions" />
</p>

| Metric | Result |
|:-------|:------:|
| Churn reduction | **14.5%** |
| 95% Confidence Interval | [7.3%, 21.8%] |
| P-value | < 0.001 ✅ |
| Customers saved per batch | ~55 |
| Annual revenue preserved | ~$42,600 |
| Campaign cost | ~$16,500 |
| **Net ROI** | **~$26,000/year** |

---

## 🛠 Methodology Pipeline

```
EDA & Cleaning → Feature Engineering → 3-Way Stratified Split (60/20/20)
    → 3-Model Comparison (LR / RF / XGBoost) with 5-Fold CV
        → SHAP Explainability (Global + Local + Segment)
            → Unit Economics & ROI Heatmap
                → Profit-Maximizing Threshold (τ=0.36)
                    → A/B Test Power Analysis & Bootstrap Simulation
                        → Go/No-Go Decision Framework
```

### Strategic Recommendations (SHAP-Driven)

| Priority | Segment | Root Cause | Intervention |
|:--------:|---------|------------|-------------|
| 🔴 1 | Month-to-month + Low tenure | Price sensitivity, no lock-in | 15–20% loyalty discount for 12-month conversion |
| 🔴 2 | Fiber optic users | 2.2× churn rate vs DSL | Service quality audit + competitive pricing review |
| 🟡 3 | No TechSupport/Security | Missing protective bundles | 3-month free trial auto-enrollment |
| 🟡 4 | Electronic check payers | Payment friction | $5/month autopay migration incentive |

---

## 🗂 Project Structure

```
churn-prediction/
├── churn prediction.ipynb    # Full analysis notebook (84 cells)
├── README.md                 # ← You are here
├── requirements.txt          # Python dependencies
├── assets/
│   ├── model_comparison.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   ├── shap_contract_segment.png
│   ├── roi_heatmap.png
│   ├── threshold_optimization.png
│   ├── feature_importance.png
│   └── ab_test_simulation.png
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/yingruma1999-hub/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
jupyter notebook "churn prediction.ipynb"
```

---

## 📌 Key Takeaway

> **This project goes beyond "build a model and report AUC."** It answers the questions stakeholders actually ask:
> - *"How much money does this save us?"* → $53K expected profit at optimal threshold
> - *"Which customers should we contact?"* → SHAP-driven segment prioritization
> - *"How do we validate this before scaling?"* → Full A/B test design with power analysis
> - *"What's the ROI?"* → ~$26K/year net, robust across cost assumptions
>
> **The complete pipeline — prediction → economics → experimentation — demonstrates production-readiness.**

---

## 📂 Data Source

> **IBM Telco Customer Churn** — synthetic dataset from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). No real customer PII.

---

## 👤 Author

**Yingru Ma** · Economics Background · Targeting DS / PM Roles

[![GitHub](https://img.shields.io/badge/GitHub-yingruma1999--hub-181717?logo=github)](https://github.com/yingruma1999-hub)
