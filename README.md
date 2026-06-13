# Credit Card Default Prediction

A two-phase machine learning project that predicts whether a credit card client will default on their next payment, using the UCI Default of Credit Card Clients Dataset.

**Group 35** — Nikhila M (S4087528), Aiswarya Sudhir (S4084978), Rajan Vipulkumar Patel (S4113210)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Phase 1 — Data Cleaning & Exploration](#phase-1--data-cleaning--exploration)
- [Phase 2 — Predictive Modelling](#phase-2--predictive-modelling)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Overview

This project builds a supervised classification pipeline to predict credit card payment default using behavioural and demographic features from 30,000 Taiwanese bank clients (April–September 2005). The target variable is whether a client will default in October 2005.

The project is structured in two phases:

- **Phase 1** — Data sourcing, cleaning, preprocessing, and exploratory data analysis (EDA)
- **Phase 2** — Feature selection, model fitting, hyperparameter tuning, and model comparison

The primary evaluation metric throughout is **ROC-AUC**, chosen because of the class imbalance in the dataset (~22% defaulters vs ~78% non-defaulters).

---

## Dataset

**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/) (Yeh & Lien, 2009)

| File | Description |
|------|-------------|
| `UCI_Credit_Card.csv` | Raw dataset with 30,000 records and 25 features (including `ID`) |
| `UCI_Credit_Card_cleaned.csv` | Cleaned dataset with 29,566 records and 24 features after preprocessing |

### Features

| Feature | Description |
|---------|-------------|
| `LIMIT_BAL` | Credit limit (NT dollars) |
| `SEX` | Gender (1 = male, 2 = female) |
| `EDUCATION` | Education level (1 = graduate, 2 = university, 3 = high school, 4 = others) |
| `MARRIAGE` | Marital status (1 = married, 2 = single, 3 = others) |
| `AGE` | Age in years |
| `PAY_0`, `PAY_2`–`PAY_6` | Repayment status for April–September 2005 |
| `BILL_AMT1`–`BILL_AMT6` | Bill statement amounts (April–September 2005) |
| `PAY_AMT1`–`PAY_AMT6` | Amount of previous payments (April–September 2005) |
| `default.payment.next.month` | Target variable (1 = default, 0 = no default) |

---

## Repository Structure

```
credit-card-default/
│
├── UCI_Credit_Card.csv              # Raw dataset
├── UCI_Credit_Card_cleaned.csv      # Cleaned dataset (Phase 1 output)
├── ML_phase1.ipynb                  # Phase 1: Data cleaning & EDA
├── Predictive_modelling.ipynb       # Phase 2: Modelling pipeline
└── README.md
```

---

## Phase 1 — Data Cleaning & Exploration

Implemented in `ML_phase1.ipynb`.

### Preprocessing Steps

1. **Removed** the `ID` column (no predictive value)
2. **Confirmed** no missing values across all features
3. **Removed** 35 duplicate rows (dataset reduced to 29,965)
4. **Removed** 399 rows with undocumented categorical values:
   - `EDUCATION` values 0, 5, 6
   - `MARRIAGE` value 0
5. **Retained** outliers in financial columns (`LIMIT_BAL`, `BILL_AMT*`, `PAY_AMT*`) — extreme values represent genuine financial behaviour and were kept intact
6. **Final cleaned dataset:** 29,566 observations × 24 features

### Key EDA Findings

- **Class imbalance:** ~22.1% defaulters vs ~77.9% non-defaulters
- **Strongest predictor:** `PAY_0` (most recent repayment status) — clients with ≥1 month payment delay were substantially more likely to default
- **Credit limit:** Defaulters had a lower median credit limit (~NT$100,000) vs non-defaulters (~NT$150,000)
- **High multicollinearity:** `BILL_AMT1`–`BILL_AMT6` are highly correlated (r ≈ 0.95)
- **Weak demographic signals:** Age, education, and sex showed weaker associations with default than behavioural features
- **Payment amounts:** Non-defaulters consistently made higher payments across all six months

---

## Phase 2 — Predictive Modelling

Implemented in `Predictive_modelling.ipynb`.

### Data Preparation

- Loaded `UCI_Credit_Card_cleaned.csv` (integer-encoded; no further cleaning needed)
- **Train/test split:** 80% training / 20% test, stratified to preserve class ratio, `random_state=42`

### Feature Selection

Two complementary methods were applied to the training set only (to prevent data leakage):

1. **SelectKBest (`f_classif`)** — ANOVA F-test ranking features by univariate association with the target
2. **Random Forest Feature Importance** — Mean decrease in impurity, capturing non-linear relationships and inter-feature dependencies

Top 15 features were selected based on agreement between both methods. Key result: `PAY_0` was the dominant feature in both rankings. The two methods differed most on the `BILL_AMT` columns (ranked higher by RF due to multicollinearity discounting) and lagged repayment columns `PAY_2`–`PAY_6` (ranked higher by `f_classif`).

### Models

Five algorithms were evaluated from different scikit-learn submodules:

| Model | Class Imbalance Handling | Notes |
|-------|--------------------------|-------|
| Logistic Regression | `class_weight='balanced'` | Linear probabilistic baseline |
| Decision Tree | `class_weight='balanced'` | Rule-based, fully interpretable |
| Random Forest | `class_weight='balanced'` | Ensemble; reduces variance via bagging |
| Gaussian Naive Bayes | `priors=[0.5, 0.5]` | Probabilistic; conditional-independence assumption |
| MLP Neural Network | *(not supported)* | Non-linear; performance frontier per literature |

### Hyperparameter Tuning

- **Method:** `GridSearchCV` with **10-fold stratified cross-validation**
- **Scoring:** ROC-AUC
- Each model's best configuration was refit on the full training set before test-set evaluation

### Model Comparison

All five tuned models were compared using 10-fold stratified cross-validation AUC scores, with **paired t-tests** (`scipy.stats.ttest_rel`) at α = 0.05 to determine statistical significance of performance differences.

---

## Results

### Cross-Validated ROC-AUC (10-fold)

| Model | Mean CV AUC | Std CV AUC |
|-------|-------------|------------|
| **Random Forest** | **0.7774** | 0.0154 |
| MLP Neural Network | 0.7671 | 0.0132 |
| Decision Tree | 0.7581 | — |
| Logistic Regression | 0.7238 | — |
| Naive Bayes | 0.7232 | — |

### Key Findings

- **Random Forest** is the best-performing model — highest mean AUC and statistically significant improvement over all four other models (p < 0.05 in every pairwise comparison)
- **MLP** ranked second with the lowest variance across folds, but its lack of `class_weight` support caused severe default recall suppression (33.6%) at the hard decision boundary
- **Logistic Regression and Naive Bayes** were statistically indistinguishable from each other (p = 0.7913), suggesting both hit a similar performance ceiling
- **Naive Bayes** achieved the highest default recall (94.6%) but at very low precision (24.2%)

### Test Set Performance Summary

| Model | Test AUC | Default Recall | Default Precision |
|-------|----------|----------------|-------------------|
| Random Forest | 0.7613 | 54.3% | 49.1% |
| MLP Neural Network | 0.7463 | 33.6% | 67.6% |
| Logistic Regression | — | 67.9% | 34.6% |
| Naive Bayes | — | 94.6% | 24.2% |

---

## Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

> **Note:** The notebooks were originally developed in **Google Colab** and use `google.colab.drive` for file paths. Update the `path` variable in `Predictive_modelling.ipynb` to match your local directory before running.

---

## How to Run

1. Clone the repository and navigate to the project folder
2. Ensure `UCI_Credit_Card.csv` is in your working directory
3. Run `ML_phase1.ipynb` to reproduce data cleaning and EDA — this outputs `UCI_Credit_Card_cleaned.csv`
4. Run `Predictive_modelling.ipynb` to reproduce feature selection, model training, tuning, and comparison

---

## References

- Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Alam, T. M., et al. (2020). An Investigation of Credit Card Default Prediction in the Imbalanced Datasets. *IEEE Access, 8*, 201173–201198.
- Butaru, F., et al. (2016). Risk and risk management in the credit card industry. *Journal of Banking & Finance, 72*, 218–239.

Full reference list available in `Predictive_modelling.ipynb` Section 5.
