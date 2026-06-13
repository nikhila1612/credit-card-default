# Credit Card Default Prediction

A two-phase machine learning project that predicts whether a credit card client will default on their next payment, using the UCI Default of Credit Card Clients Dataset.

**Group 35** | Nikhila M (S4087528), Aiswarya Sudhir (S4084978), Rajan Vipulkumar Patel (S4113210)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Phase 1: Data Cleaning and Exploration](#phase-1-data-cleaning-and-exploration)
- [Phase 2: Predictive Modelling](#phase-2-predictive-modelling)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Overview

This project explores whether we can predict credit card payment defaults before they happen. We worked with a real-world dataset of 30,000 Taiwanese bank clients, covering their repayment behaviour from April to September 2005, and trained models to predict whether each client would default in October 2005.

The work is split into two phases. Phase 1 focused on getting the data into a trustworthy shape and understanding what it was telling us. Phase 2 took those insights and built five machine learning models, tuned each one carefully, and compared them to find the best approach.

We used **ROC-AUC** as our main evaluation metric throughout, since the dataset is imbalanced (roughly 22% defaulters vs 78% non-defaulters) and accuracy alone would have been misleading.

---

## Dataset

**Source:** [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (Yeh & Lien, 2009)

| File | Description |
|------|-------------|
| `UCI_Credit_Card.csv` | The original raw dataset with 30,000 records and 25 columns |
| `UCI_Credit_Card_cleaned.csv` | The cleaned version used for modelling, with 29,566 records and 24 columns |

### Features

| Feature | Description |
|---------|-------------|
| `LIMIT_BAL` | Credit limit in NT dollars |
| `SEX` | Gender (1 = male, 2 = female) |
| `EDUCATION` | Education level (1 = graduate, 2 = university, 3 = high school, 4 = others) |
| `MARRIAGE` | Marital status (1 = married, 2 = single, 3 = others) |
| `AGE` | Age in years |
| `PAY_0`, `PAY_2` to `PAY_6` | Repayment status for April to September 2005 |
| `BILL_AMT1` to `BILL_AMT6` | Monthly bill statement amounts |
| `PAY_AMT1` to `PAY_AMT6` | Amount paid each month |
| `default.payment.next.month` | Target variable (1 = defaulted, 0 = did not default) |

---

## Repository Structure

```
credit-card-default/
├── UCI_Credit_Card.csv              # Raw dataset
├── UCI_Credit_Card_cleaned.csv      # Cleaned dataset output from Phase 1
├── ML_phase1.ipynb                  # Phase 1: Data cleaning and EDA
├── Predictive_modelling.ipynb       # Phase 2: Modelling pipeline
└── README.md
```

---

## Phase 1: Data Cleaning and Exploration

Notebook: `ML_phase1.ipynb`

### What we cleaned

We started with 30,000 records and worked through the data systematically before touching any modelling.

1. Dropped the `ID` column since it carries no useful information
2. Confirmed there were no missing values anywhere in the dataset
3. Removed 35 duplicate rows, bringing the count to 29,965
4. Removed 399 rows where `EDUCATION` had values of 0, 5, or 6, and where `MARRIAGE` had a value of 0 — none of these appear in the dataset documentation, so rather than guessing what category they belonged to, we removed them entirely
5. Kept all outliers in the financial columns (`LIMIT_BAL`, `BILL_AMT*`, `PAY_AMT*`) since extreme values here reflect genuine client behaviour, not data errors

The final cleaned dataset has **29,566 rows and 24 features**.

### What the data told us

A few things stood out clearly from the exploratory analysis:

The dataset is imbalanced — only about 22% of clients defaulted. This meant we had to think carefully about our evaluation approach from the start.

The single strongest predictor of default was `PAY_0`, the most recent repayment status. Clients who were even one month behind on payments were significantly more likely to default the following month.

Clients who defaulted tended to have lower credit limits (around NT$100,000 median) compared to those who didn't (around NT$150,000). Meanwhile, the six monthly bill amount columns (`BILL_AMT1` through `BILL_AMT6`) turned out to be very highly correlated with each other (r ≈ 0.95), which created a multicollinearity problem we had to address in Phase 2. Demographic features like age, sex, and education were far weaker predictors than the behavioural ones.

---

## Phase 2: Predictive Modelling

Notebook: `Predictive_modelling.ipynb`

### Setup

We loaded the cleaned dataset and split it 80/20 into training and test sets, using a stratified split to keep the class ratio consistent in both halves. The test set was set aside and only used for final evaluation.

### Feature Selection

We used two different feature selection methods, both applied only to the training data:

**SelectKBest (f_classif)** scores each feature individually using an ANOVA F-test, measuring how strongly its distribution differs between defaulters and non-defaulters. This is fast and interpretable but treats each feature in isolation.

**Random Forest Importance** evaluates features together, which means it naturally discounts redundant ones. This was particularly useful for dealing with the correlated `BILL_AMT` columns that the univariate method couldn't differentiate between.

We selected the top 15 features based on where both methods agreed. `PAY_0` was the clear standout in both rankings. The methods diverged most on the bill amount columns and lagged repayment statuses, which is exactly the multicollinearity pattern we expected from Phase 1.

### Models

We trained five models, each from a different scikit-learn submodule to keep the comparison methodologically diverse:

| Model | How we handled class imbalance |
|-------|-------------------------------|
| Logistic Regression | `class_weight='balanced'` |
| Decision Tree | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| Gaussian Naive Bayes | `priors=[0.5, 0.5]` |
| MLP Neural Network | Not supported (limitation noted) |

### Tuning

Each model was tuned using `GridSearchCV` with 10-fold stratified cross-validation, optimising for ROC-AUC. The best configuration from the grid search was then refit on the full training set before being tested.

### Comparison

To compare models fairly, we ran 10-fold stratified cross-validation across all five and used paired t-tests (`scipy.stats.ttest_rel`) to check whether any performance differences were statistically significant rather than just noise.

---

## Results

### Cross-Validated ROC-AUC (10-fold)

| Model | Mean CV AUC | Std |
|-------|-------------|-----|
| **Random Forest** | **0.7774** | 0.0154 |
| MLP Neural Network | 0.7671 | 0.0132 |
| Decision Tree | 0.7581 | |
| Logistic Regression | 0.7238 | |
| Naive Bayes | 0.7232 | |

### What we found

Random Forest came out on top with a mean AUC of 0.7774 and was statistically significantly better than every other model (p < 0.05 in all pairwise comparisons). Its ensemble approach handled the non-linear relationships in the data better than any single model could.

MLP placed second and was the most consistent across folds, but without `class_weight` support in scikit-learn, it ended up biased toward the majority class, catching only 33.6% of actual defaulters on the test set despite its strong AUC.

Logistic Regression and Naive Bayes performed almost identically (p = 0.7913), suggesting both models hit the same ceiling on this dataset despite being very different in how they work.

### Test Set Performance

| Model | Test AUC | Default Recall | Default Precision |
|-------|----------|----------------|-------------------|
| Random Forest | 0.7613 | 54.3% | 49.1% |
| MLP Neural Network | 0.7463 | 33.6% | 67.6% |
| Logistic Regression | | 67.9% | 34.6% |
| Naive Bayes | | 94.6% | 24.2% |

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

Install everything with:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

> These notebooks were originally written in Google Colab. If you're running them locally, update the `path` variable in `Predictive_modelling.ipynb` to point to your local directory.

---

## How to Run

1. Clone the repository and open the project folder
2. Make sure `UCI_Credit_Card.csv` is in your working directory
3. Run `ML_phase1.ipynb` first — this cleans the data and saves `UCI_Credit_Card_cleaned.csv`
4. Then run `Predictive_modelling.ipynb` to reproduce the full modelling pipeline

---

## References

- Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Alam, T. M., et al. (2020). An Investigation of Credit Card Default Prediction in the Imbalanced Datasets. *IEEE Access, 8*, 201173–201198.
- Butaru, F., et al. (2016). Risk and risk management in the credit card industry. *Journal of Banking & Finance, 72*, 218–239.

The full reference list is in Section 5 of `Predictive_modelling.ipynb`.
