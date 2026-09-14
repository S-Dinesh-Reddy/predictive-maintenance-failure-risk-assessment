# Machine Learning-Based Predictive Maintenance and Failure Risk Assessment

A machine learning project for predicting machine failure risk from industrial equipment operating conditions using the **AI4I 2020 Predictive Maintenance Dataset**.

The project develops and evaluates multiple machine learning classification models and deploys the selected model through a Flask-based web application for real-time failure risk assessment.

---

## 📌 Project Overview

Unexpected machine failures can lead to production downtime, maintenance costs, and equipment damage.

This project aims to develop a machine learning-based predictive maintenance system that analyzes machine operating parameters and predicts whether a machine is at risk of failure.

The complete workflow includes:

**Data → Preprocessing → EDA → Feature Engineering → Class Imbalance Handling → Train/Test Preparation → Model Training → Evaluation → Hyperparameter Tuning → Flask Deployment**

---

## 🎯 Objectives

- Analyze industrial machine operating data.
- Clean and preprocess the dataset for machine learning.
- Perform exploratory data analysis (EDA).
- Engineer meaningful features from machine parameters.
- Analyze and address the highly imbalanced failure-class distribution.
- Prepare training and testing datasets.
- Train and compare multiple machine learning classification models.
- Evaluate models using suitable classification metrics.
- Tune the best-performing model.
- Deploy the final model using Flask.
- Provide a web interface for real-time machine failure risk assessment.

---

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

The dataset contains simulated industrial machine operating conditions and failure information.

### Original Dataset

- **Records:** 10,000
- **Columns:** 14
- **Target:** `Machine failure`

### Main Parameters

- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- Machine Type
- Machine Failure

The target variable is highly imbalanced:

- **No Failure:** 9,661 samples (96.61%)
- **Failure:** 339 samples (3.39%)

Because failure cases are relatively rare, evaluation focuses strongly on **Precision, Recall, F1-score, and ROC-AUC**, rather than accuracy alone.

---

## 🔄 Project Workflow

### Phase 1 — Data Preprocessing

- Dataset loading and validation
- Data cleaning
- Identifier removal
- Machine Type encoding
- Feature engineering
- Class imbalance analysis
- Stratified train/test split
- Feature scaling
- Stratified 5-fold cross-validation preparation

### Phase 2 — Model Training and Evaluation

Five classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. XGBoost

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Phase 3 — Model Tuning

XGBoost was selected as the final model candidate and tuned using **RandomizedSearchCV with Stratified 5-fold Cross-Validation**.

The tuned model achieved a significant improvement in test F1-score compared with the original XGBoost model.

### Phase 4 — Deployment

The final tuned XGBoost model was integrated into a **Flask web application**.

The application accepts machine operating parameters and returns:

- Failure prediction
- Failure probability
- Risk level
- Maintenance recommendation

---

## ⚙️ Feature Engineering

Two additional features were created from the original machine parameters.

### Temperature Difference

```text
Temperature difference = Process temperature − Air temperature