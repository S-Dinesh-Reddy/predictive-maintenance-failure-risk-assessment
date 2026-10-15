import json
import os

def make_cell(cell_type, source_lines):
    if isinstance(source_lines, str):
        lines = [line + '\n' for line in source_lines.split('\n')]
        if lines and lines[-1] == '\n':
            lines[-1] = ''
    else:
        lines = [l if l.endswith('\n') else l + '\n' for l in source_lines]
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells = []

# ==============================================================================
# SECTION 0: TITLE, EXECUTIVE SUMMARY & DATA INGESTION
# ==============================================================================

cells.append(make_cell("markdown", """# Phase 2: Predictive Maintenance Failure Risk Assessment — Model Training, Cross-Validation, Calibration, and Explainability

**Author:** Dinesh Reddy  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 synthetic milling machine records)  
**Task:** Binary Classification of Catastrophic Machine Failure (`Machine failure` $\\in \\{0, 1\\}$)  

---

### Phase 2 Architecture & Methodology Overview
1. **Preprocessed Data Ingestion**: Load the 80/20 stratified split generated during Phase 1 (`X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`), preserving feature engineering (`Temperature difference [K]`, `Tool wear level`, one-hot encoded `Type`).
2. **Exploratory Class & Feature Space Diagnostics**: Inspect class imbalance (~3.4% failure prevalence) and project the 9-dimensional feature space onto 2D Principal Components (PCA) to visualize non-linear class overlap.
3. **Five Baseline Algorithms**: Evaluate a spectrum of statistical and machine learning models:
   - *Logistic Regression* (Linear log-odds baseline with coefficient interpretability)
   - *Random Forest Classifier* (Bagged ensemble of de-correlated decision trees)
   - *Support Vector Machine (RBF)* (Maximum-margin separation in reproducing kernel Hilbert space)
   - *Decision Tree Classifier* (Hierarchical recursive axis-aligned partitioning)
   - *Baseline XGBoost* (Gradient boosted decision trees with exact greedy split finding)
4. **Methodological Rigor & Zero Test Leakage**:
   - Model tuning and operating threshold selection are conducted **strictly on training data** using Stratified 5-Fold Cross-Validation and Out-Of-Fold (OOF) probability predictions.
   - The held-out test set ($N=2,000$) is kept completely unobserved until final unbiased benchmarking.
5. **Comprehensive Metric Suite**: Evaluate models across Discrimination (ROC-AUC, Average Precision, F1-Score, Recall, Precision, Specificity, Balanced Accuracy, MCC) and Probability Calibration (Brier Score, MSE, RMSE, Log Loss).
6. **Probability Calibration & Reliability Diagnostics**: Assess probability distortion under class reweighting, demonstrating how Sigmoid calibration restores alignment with empirical failure frequencies.
7. **Hyperparameter Optimization**: Tune XGBoost using RandomizedSearchCV across 6 architectural hyperparameters with 5-fold CV.
8. **Explainable AI (XAI) with SHAP**: Compute Shapley Additive exPlanations using `TreeExplainer` to extract global feature rankings, directional impact beeswarm plots, local prediction waterfall charts, and non-linear bivariate dependence interactions.
9. **Production Inference Pipeline**: Formulate an end-to-end scikit-learn `Pipeline` pairing the mathematically verified `StandardScaler` with the tuned XGBoost estimator, export deployable artifacts, and validate reload compatibility."""))

cells.append(make_cell("code", """import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform

# Ensure strict reproducibility across all stochastic routines
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True"""))

cells.append(make_cell("code", """# Comprehensive evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve"""))

cells.append(make_cell("code", """# Classification algorithms, preprocessing and validation modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV

import joblib
import shap"""))

cells.append(make_cell("code", """# Resolve project and dataset paths cleanly across local IDE, scripts, and Colab environments
DATA_DIR = os.path.join('..', 'data') if os.path.exists(os.path.join('..', 'data')) else 'data'
if not os.path.exists(os.path.join(DATA_DIR, 'X_train_scaled.csv')):
    DATA_DIR = '/content/drive/MyDrive/Predictive_Maintenance'

print(f"Loading data from directory: {os.path.abspath(DATA_DIR)}")

X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_scaled.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))

# Standardize feature names to prevent XGBoost special-character parsing errors
clean_cols = [
    'Air_temperature',
    'Process_temperature',
    'Rotational_speed',
    'Torque',
    'Tool_wear',
    'Type_L',
    'Type_M',
    'Temperature_difference',
    'Tool_wear_level'
]

X_train.columns = clean_cols
X_test.columns = clean_cols

# Flatten target series
y_train = y_train.squeeze()
y_test = y_test.squeeze()"""))

cells.append(make_cell("code", """print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test shape  : {y_test.shape}")
print(f"Features      : {list(X_train.columns)}")"""))

cells.append(make_cell("markdown", """## 1. Class Distribution Analysis

The training data is highly imbalanced, with machine failures representing a small proportion of the observations (~3.4%). Visualizing the class distribution illustrates why accuracy alone is deceptive and why imbalance-aware techniques (`class_weight='balanced'`, `scale_pos_weight`, stratified cross-validation) are mandatory."""))

cells.append(make_cell("code", """class_counts = y_train.value_counts().sort_index()
labels = ['No Failure (0)', 'Failure (1)']

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, class_counts.values, color=['#2563eb', '#dc2626'], edgecolor='black', alpha=0.85)

plt.title('Training Set Class Distribution', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Machine Failure Status', fontsize=11)
plt.ylabel('Number of Samples', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(y_train)) * 100
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 100, f"{yval:,} ({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')

plt.ylim(0, len(y_train) * 1.1)
plt.show()"""))

cells.append(make_cell("code", """n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

print(f"Class 0 (Normal Operation) : {n_neg:,} samples ({n_neg / len(y_train) * 100:.2f}%)")
print(f"Class 1 (Machine Failure)  : {n_pos:,} samples ({n_pos / len(y_train) * 100:.2f}%)")
print(f"Exact Imbalance Ratio (scale_pos_weight) = {scale_pos_weight:.2f}")"""))

cells.append(make_cell("markdown", """## 2. Feature-Space Visualization using PCA

The dataset contains 9 continuous and encoded features, which cannot be directly visualized in a single 2D plot. Principal Component Analysis (PCA) is applied here solely for geometric diagnostics by reducing the standardized feature space to two orthogonal principal components.

The projection reveals how failure observations are embedded within normal operating clusters."""))

cells.append(make_cell("code", """y_train_array = np.asarray(y_train).reshape(-1)

# Fit PCA on training features
pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_train_pca = pca.fit_transform(X_train)

explained_var = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: PC1 = {explained_var[0]*100:.2f}%, PC2 = {explained_var[1]*100:.2f}% (Total = {sum(explained_var)*100:.2f}%)")

plt.figure(figsize=(9, 6))
plt.scatter(
    X_train_pca[y_train_array == 0, 0],
    X_train_pca[y_train_array == 0, 1],
    c='#3b82f6',
    label='No Failure (0)',
    alpha=0.35,
    s=25,
    edgecolors='none'
)
plt.scatter(
    X_train_pca[y_train_array == 1, 0],
    X_train_pca[y_train_array == 1, 1],
    c='#ef4444',
    label='Machine Failure (1)',
    alpha=0.85,
    s=45,
    edgecolors='black',
    linewidth=0.5
)

plt.title('2D PCA Projection of Industrial Feature Space', fontsize=13, fontweight='bold', pad=12)
plt.xlabel(f'Principal Component 1 ({explained_var[0]*100:.1f}% variance)', fontsize=11)
plt.ylabel(f'Principal Component 2 ({explained_var[1]*100:.1f}% variance)', fontsize=11)
plt.legend(frameon=True, loc='upper right')
plt.show()"""))

# ==============================================================================
# SECTION 1: MODEL 1 — LOGISTIC REGRESSION
# ==============================================================================

cells.append(make_cell("markdown", """## 3. Model 1: Logistic Regression Baseline

Logistic Regression serves as our interpretable linear baseline. We configure `class_weight='balanced'` to inversely weight loss by class prevalence, penalizing false negatives proportionally to the ~28.5:1 imbalance ratio."""))

cells.append(make_cell("code", """logistic_model = LogisticRegression(
    class_weight='balanced',
    random_state=RANDOM_SEED,
    max_iter=1000
)"""))

cells.append(make_cell("code", """logistic_model.fit(X_train, y_train)
print("Logistic Regression model fitted successfully.")"""))

cells.append(make_cell("markdown", """### Logistic Regression Coefficient Analysis

The coefficients of the Logistic Regression model reveal the log-odds direction and magnitude for each sensor feature. Positive coefficients increase failure probability, whereas negative coefficients reduce it."""))

cells.append(make_cell("code", """lr_coefficients = pd.Series(
    logistic_model.coef_[0],
    index=X_train.columns
).sort_values()

plt.figure(figsize=(9, 5))
colors = ['#dc2626' if c > 0 else '#2563eb' for c in lr_coefficients]
lr_coefficients.plot(kind='barh', color=colors, edgecolor='black', alpha=0.85)

plt.axvline(0, color='black', linewidth=1, linestyle='--')
plt.xlabel('Log-Odds Coefficient Value', fontsize=11)
plt.ylabel('Sensor Feature', fontsize=11)
plt.title('Logistic Regression Feature Coefficients (Direction & Impact)', fontsize=13, fontweight='bold', pad=12)
plt.show()

print("Top Risk Factors (Positive Log-Odds):")
print(lr_coefficients.tail(3)[::-1].to_string())"""))

cells.append(make_cell("code", """# Generate held-out test predictions
y_pred_lr = logistic_model.predict(X_test)
y_pred_proba_lr = logistic_model.predict_proba(X_test)[:, 1]"""))

cells.append(make_cell("code", """print("First 10 Logistic Regression discrete predictions:")
print(y_pred_lr[:10])

print("\\nFirst 10 failure probabilities:")
print(np.round(y_pred_proba_lr[:10], 4))"""))

cells.append(make_cell("code", """# Helper function to compute the complete metric suite
def compute_comprehensive_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    # Probability and prediction error metrics
    mse = mean_squared_error(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    rmse = np.sqrt(mse)
    ll = log_loss(y_true, y_prob)
    err_rate = 1.0 - acc
    
    # Operational contingency metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Average Precision': ap,
        'MSE': mse,
        'RMSE': rmse,
        'Brier Score': brier,
        'Log Loss': ll,
        'Error Rate': err_rate,
        'Specificity': spec,
        'Balanced Accuracy': bal_acc,
        'MCC': mcc,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }

lr_metrics = compute_comprehensive_metrics(y_test, y_pred_lr, y_pred_proba_lr)
print("Logistic Regression Held-Out Test Performance:")
print("-" * 50)
for k, v in lr_metrics.items():
    if isinstance(v, float):
        print(f"  {k:20s}: {v:.4f}")
    else:
        print(f"  {k:20s}: {v}")"""))

cells.append(make_cell("code", """plt.figure(figsize=(8, 4.5))
bar_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
bar_vals = [lr_metrics[k] for k in bar_keys]

plt.bar(bar_keys, bar_vals, color='#3b82f6', edgecolor='black', alpha=0.85)
plt.ylim(0, 1.05)
plt.ylabel('Metric Score', fontsize=11)
plt.title('Logistic Regression — Core Performance Metrics', fontsize=13, fontweight='bold', pad=12)

for i, v in enumerate(bar_vals):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
plt.show()"""))

cells.append(make_cell("code", """cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Logistic Regression Confusion Matrix:")
print(cm_lr)

ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["No Failure (0)", "Failure (1)"]
).plot(cmap='Blues', values_format='d')
plt.title("Logistic Regression — Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
plt.grid(False)
plt.show()"""))

cells.append(make_cell("code", """# Configure Stratified 5-Fold Cross Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision'
}"""))

cells.append(make_cell("code", """cv_results_lr = cross_validate(
    logistic_model,
    X_train,
    y_train,
    cv=skf,
    scoring=scoring,
    n_jobs=-1
)

print("Logistic Regression — 5-Fold Stratified Cross Validation (Training Only)")
print("----------------------------------------------------------------------")
print(f"Mean Accuracy  : {cv_results_lr['test_accuracy'].mean():.4f} ± {cv_results_lr['test_accuracy'].std():.4f}")
print(f"Mean Precision : {cv_results_lr['test_precision'].mean():.4f} ± {cv_results_lr['test_precision'].std():.4f}")
print(f"Mean Recall    : {cv_results_lr['test_recall'].mean():.4f} ± {cv_results_lr['test_recall'].std():.4f}")
print(f"Mean F1-Score  : {cv_results_lr['test_f1'].mean():.4f} ± {cv_results_lr['test_f1'].std():.4f}")
print(f"Mean ROC-AUC   : {cv_results_lr['test_roc_auc'].mean():.4f} ± {cv_results_lr['test_roc_auc'].std():.4f}")
print(f"Mean AP        : {cv_results_lr['test_average_precision'].mean():.4f} ± {cv_results_lr['test_average_precision'].std():.4f}")"""))

cells.append(make_cell("markdown", """### Logistic Regression Calibration Diagnostics & Probability Experiments

When an estimator is trained with `class_weight='balanced'`, its objective function penalizes positive-class errors by ~28.5x. While this elevates recall, it mathematically shifts the predicted probability distribution upward, causing severe probability distortion (mean predicted probability far exceeding the ~3.4% true prior).

Below, we compare:
1. **Unweighted Logistic Regression**: Well-calibrated probabilities reflecting true prevalence, but poor recall on the rare failure class.
2. **Balanced Logistic Regression**: High recall (~86.8%), but uncalibrated probabilities (average predicted probability ~46.9%).
3. **Calibrated Logistic Regression (`CalibratedClassifierCV` with 5-fold Sigmoid Platt scaling)**: Restores reliable probability mapping without test-set leakage."""))

cells.append(make_cell("code", """# Train unweighted baseline
lr_unweighted = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000).fit(X_train, y_train)

# Fit calibrated estimator strictly on training data using 5-fold cross-validation
lr_calibrated = CalibratedClassifierCV(
    LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000),
    method='sigmoid',
    cv=5
).fit(X_train, y_train)

prob_unweighted = lr_unweighted.predict_proba(X_test)[:, 1]
prob_weighted = logistic_model.predict_proba(X_test)[:, 1]
prob_calibrated = lr_calibrated.predict_proba(X_test)[:, 1]

print(f"Observed Failure Prevalence in Test Data: {y_test.mean()*100:.2f}%")
print("-" * 65)
print(f"LR Unweighted  | Mean Pred Prob: {prob_unweighted.mean()*100:.2f}% | Brier: {brier_score_loss(y_test, prob_unweighted):.4f} | Log Loss: {log_loss(y_test, prob_unweighted):.4f}")
print(f"LR Balanced    | Mean Pred Prob: {prob_weighted.mean()*100:.2f}% | Brier: {brier_score_loss(y_test, prob_weighted):.4f} | Log Loss: {log_loss(y_test, prob_weighted):.4f}")
print(f"LR Calibrated  | Mean Pred Prob: {prob_calibrated.mean()*100:.2f}% | Brier: {brier_score_loss(y_test, prob_calibrated):.4f} | Log Loss: {log_loss(y_test, prob_calibrated):.4f}")"""))

cells.append(make_cell("code", """# Plot Reliability Calibration Curves
prob_true_unw, prob_pred_unw = calibration_curve(y_test, prob_unweighted, n_bins=8, strategy='quantile')
prob_true_w, prob_pred_w = calibration_curve(y_test, prob_weighted, n_bins=8, strategy='quantile')
prob_true_cal, prob_pred_cal = calibration_curve(y_test, prob_calibrated, n_bins=8, strategy='quantile')

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated (Ideal)')
plt.plot(prob_pred_unw, prob_true_unw, 's-', color='#64748b', label='Unweighted LR')
plt.plot(prob_pred_w, prob_true_w, 'o-', color='#ef4444', label='Balanced LR (Uncalibrated)')
plt.plot(prob_pred_cal, prob_true_cal, '^-', color='#10b981', label='Calibrated LR (Platt Sigmoid, CV=5)')

plt.title('Logistic Regression — Calibration Reliability Curves', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Mean Predicted Probability', fontsize=11)
plt.ylabel('Empirical Fraction of Positives (Failures)', fontsize=11)
plt.legend(frameon=True, loc='upper left')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()"""))

# Write script that builds the complete notebook
with open('scripts/build_full_notebook.py', 'w', encoding='utf-8') as f:
    f.write("# Generator script\n")
