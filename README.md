# Machine Learning-Based Predictive Maintenance and Failure Risk Assessment

A machine learning project for predicting machine failure risk from industrial equipment operating conditions using the **AI4I 2020 Predictive Maintenance Dataset**.

## 📌 Project Overview

Unexpected machine failures can lead to production downtime, maintenance costs, and equipment damage. This project aims to develop a machine learning-based predictive maintenance system that analyzes machine operating parameters and predicts the risk of failure.

The project follows a complete ML workflow:

**Data → Preprocessing → EDA → Feature Engineering → Class Imbalance Handling → Train/Test Preparation → Model Training → Evaluation → Explainability → Deployment**

## 🎯 Objectives

- Analyze industrial machine operating data.
- Clean and preprocess the dataset for machine learning.
- Perform exploratory data analysis (EDA).
- Engineer meaningful features from machine parameters.
- Address the highly imbalanced failure-class distribution.
- Prepare training and testing datasets.
- Train and compare suitable machine learning classification models.
- Evaluate model performance using appropriate classification metrics.
- Provide model interpretability using SHAP.
- Integrate the trained model into a Flask-based web application.

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

The dataset contains machine operating conditions and failure information for industrial equipment.

### Main Parameters

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type
- Machine Failure

The original dataset contains **10,000 records and 14 columns**.

## 🔄 Project Workflow

1. **AI4I 2020 Dataset**
   ↓
2. **Data Cleaning & Validation**
   ↓
3. **Exploratory Data Analysis (EDA)**
   ↓
4. **Feature Engineering**
   ↓
5. **Class Imbalance Analysis**
   ↓
6. **Train/Test Split**
   ↓
7. **Feature Scaling**
   ↓
8. **Stratified Cross-Validation**
   ↓
9. **Model Training** 🔄
   ↓
10. **Model Evaluation** ⏳
   ↓
11. **SHAP Explainability** ⏳
   ↓
12. **Flask Web Application** ⏳
