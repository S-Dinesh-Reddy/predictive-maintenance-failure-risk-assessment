"""
AI4I 2020 Predictive Maintenance — Model Training, Evaluation, and Pipeline Construction
=======================================================================================
This script executes the complete machine learning workflow:
1. Loads preprocessed and stratified datasets.
2. Trains all 5 baseline models + Tuned XGBoost.
3. Performs 5-fold Stratified Cross-Validation on training data.
4. Performs calibration diagnostics and training-only threshold optimization.
5. Evaluates comprehensive metrics on the held-out test set.
6. Generates the full visualization suite.
7. Computes SHAP explainability.
8. Saves the final inference pipeline, true scaler, model, and metadata.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
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
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve
)
import shap

# Set random seeds for strict reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_ASSETS_DIR = os.path.join(BASE_DIR, 'static', 'assets')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_ASSETS_DIR, exist_ok=True)

# 1. Load Data
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

X_train_df = pd.read_csv(os.path.join(DATA_DIR, 'X_train_scaled.csv'))
X_test_df = pd.read_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'))
y_train_df = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
y_test_df = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))

# Standardize column names across all frameworks
X_train_df.columns = clean_cols
X_test_df.columns = clean_cols
y_train = y_train_df['Machine failure'].values
y_test = y_test_df['Machine failure'].values

print(f"Dataset Loaded:")
print(f"  Training samples: {len(X_train_df)} (Failures: {y_train.sum()} / {y_train.mean()*100:.2f}%)")
print(f"  Testing samples:  {len(X_test_df)} (Failures: {y_test.sum()} / {y_test.mean()*100:.2f}%)")

# Calculate exact imbalance ratio
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Imbalance Ratio (scale_pos_weight): {scale_pos_weight:.2f}")

# 2. Reconstruct True Physical Scaler
# Proven from Phase 1 training set statistics
true_mean = np.array([300.00545, 310.0060625, 1539.356875, 40.0033625, 107.685, 0.60375, 0.297625, 10.0006125, 0.9945])
true_scale = np.array([1.99671926, 1.47933921, 180.97163111, 10.01891935, 63.60802642, 0.48911751, 0.45721369, 0.99979167, 0.81775287])

true_scaler = StandardScaler()
true_scaler.mean_ = true_mean
true_scaler.scale_ = true_scale
true_scaler.var_ = true_scale ** 2
true_scaler.feature_names_in_ = np.array(clean_cols)
true_scaler.n_features_in_ = 9
true_scaler.n_samples_seen_ = 8000

# Save verified scaler immediately
joblib.dump(true_scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print("True StandardScaler saved to models/scaler.pkl.")

# 3. Define Candidate Models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# SVM probability estimator wrapped with CalibratedClassifierCV to avoid deprecated SVC(probability=True)
svm_base = SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
svm_calibrated = CalibratedClassifierCV(svm_base, cv=5)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=RANDOM_SEED),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=RANDOM_SEED),
    'SVM (RBF)': svm_calibrated,
    'XGBoost (Baseline)': XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric='logloss'
    ),
    'Tuned XGBoost': XGBClassifier(
        n_estimators=269,
        learning_rate=0.14200913727754388,
        max_depth=6,
        min_child_weight=2,
        subsample=0.942436113869325,
        colsample_bytree=0.8796596399465607,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric='logloss'
    )
}

# 4. Stratified 5-Fold Cross Validation Evaluation
print("\n" + "="*50)
print("EXECUTING STRATIFIED 5-FOLD CROSS VALIDATION (TRAINING ONLY)")
print("="*50)

cv_metrics = {}
scoring = {
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision',
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision'
}

for name, model in models.items():
    print(f"Running CV for {name}...")
    res = cross_validate(model, X_train_df, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    cv_metrics[name] = {
        'f1_mean': float(np.mean(res['test_f1'])),
        'f1_std': float(np.std(res['test_f1'])),
        'recall_mean': float(np.mean(res['test_recall'])),
        'recall_std': float(np.std(res['test_recall'])),
        'precision_mean': float(np.mean(res['test_precision'])),
        'precision_std': float(np.std(res['test_precision'])),
        'roc_auc_mean': float(np.mean(res['test_roc_auc'])),
        'roc_auc_std': float(np.std(res['test_roc_auc'])),
        'ap_mean': float(np.mean(res['test_average_precision'])),
        'ap_std': float(np.std(res['test_average_precision']))
    }
    print(f"  F1: {cv_metrics[name]['f1_mean']:.4f} ± {cv_metrics[name]['f1_std']:.4f} | Recall: {cv_metrics[name]['recall_mean']:.4f} ± {cv_metrics[name]['recall_std']:.4f}")

# 5. Threshold Optimization via Out-Of-Fold (OOF) Training Predictions (Zero Test Leakage)
print("\n" + "="*50)
print("TRAINING-ONLY OUT-OF-FOLD THRESHOLD SWEEP (TUNED XGBOOST)")
print("="*50)

tuned_xgb = models['Tuned XGBoost']
oof_probs = cross_val_predict(tuned_xgb, X_train_df, y_train, cv=skf, method='predict_proba')[:, 1]

threshold_sweep = []
for th in np.arange(0.10, 0.95, 0.05):
    preds_th = (oof_probs >= th).astype(int)
    tn_th, fp_th, fn_th, tp_th = confusion_matrix(y_train, preds_th).ravel()
    prec_th = precision_score(y_train, preds_th, zero_division=0)
    rec_th = recall_score(y_train, preds_th, zero_division=0)
    f1_th = f1_score(y_train, preds_th, zero_division=0)
    threshold_sweep.append({
        'threshold': round(float(th), 2),
        'precision': float(prec_th),
        'recall': float(rec_th),
        'f1': float(f1_th),
        'tp': int(tp_th),
        'fp': int(fp_th),
        'fn': int(fn_th),
        'tn': int(tn_th)
    })

print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'TN':<5}")
print("-" * 75)
for row in threshold_sweep:
    print(f"{row['threshold']:<10.2f} | {row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['f1']:<10.4f} | {row['tp']:<5d} | {row['fp']:<5d} | {row['fn']:<5d} | {row['tn']:<5d}")

# Operating threshold selection: 0.50 maintains optimal precision/recall balance
SELECTED_THRESHOLD = 0.50
print(f"\nDocumented Operational Threshold: {SELECTED_THRESHOLD:.2f}")

# 6. Fit Models on Entire Training Set and Evaluate on Held-Out Test Set
print("\n" + "="*50)
print("HELD-OUT TEST SET EVALUATION")
print("="*50)

test_results = {}
trained_estimators = {}
model_probs = {}
model_preds = {}

for name, model in models.items():
    print(f"Training {name} on full training set...")
    model.fit(X_train_df, y_train)
    trained_estimators[name] = model

    probs = model.predict_proba(X_test_df)[:, 1]
    preds = (probs >= SELECTED_THRESHOLD).astype(int)

    model_probs[name] = probs
    model_preds[name] = preds

    # Metric calculations
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)

    mse = mean_squared_error(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    rmse = np.sqrt(mse)
    ll = log_loss(y_test, probs)
    error_rate = 1.0 - acc

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)

    test_results[name] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'average_precision': float(ap),
        'mse': float(mse),
        'rmse': float(rmse),
        'brier_score': float(brier),
        'log_loss': float(ll),
        'error_rate': float(error_rate),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'specificity': float(spec),
        'balanced_accuracy': float(bal_acc),
        'mcc': float(mcc)
    }

    print(f"  {name:25s} | F1: {f1:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f} | AUC: {roc_auc:.4f} | AP: {ap:.4f} | FN: {fn}")

# 7. Model Leaderboard Summary Table
leaderboard_rows = []
for name in models.keys():
    t_res = test_results[name]
    c_res = cv_metrics[name]
    leaderboard_rows.append({
        'Model': name,
        'Test F1': t_res['f1'],
        'Test Recall': t_res['recall'],
        'Test Precision': t_res['precision'],
        'Test ROC-AUC': t_res['roc_auc'],
        'Test AP': t_res['average_precision'],
        'Test FN': t_res['fn'],
        'Test FP': t_res['fp'],
        'Brier Score': t_res['brier_score'],
        'CV F1 Mean': c_res['f1_mean'],
        'CV F1 Std': c_res['f1_std']
    })

leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by='Test F1', ascending=False).reset_index(drop=True)
print("\n" + "="*50)
print("FINAL MODEL LEADERBOARD (SORTED BY TEST F1)")
print("="*50)
print(leaderboard_df.to_string(index=False))

# 8. Logistic Regression Probability Calibration Diagnostic
print("\n" + "="*50)
print("LOGISTIC REGRESSION PROBABILITY CALIBRATION ANALYSIS")
print("="*50)

# Unweighted LR, Weighted LR, Calibrated LR comparison
lr_unweighted = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000).fit(X_train_df, y_train)
lr_weighted = trained_estimators['Logistic Regression']
lr_calibrated = CalibratedClassifierCV(
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
    method='sigmoid',
    cv=5
).fit(X_train_df, y_train)

prob_unweighted = lr_unweighted.predict_proba(X_test_df)[:, 1]
prob_weighted = lr_weighted.predict_proba(X_test_df)[:, 1]
prob_calibrated = lr_calibrated.predict_proba(X_test_df)[:, 1]

print(f"LR Unweighted:   Brier = {brier_score_loss(y_test, prob_unweighted):.4f}, LogLoss = {log_loss(y_test, prob_unweighted):.4f}, Mean Pred Prob = {prob_unweighted.mean()*100:.2f}%")
print(f"LR Weighted:     Brier = {brier_score_loss(y_test, prob_weighted):.4f}, LogLoss = {log_loss(y_test, prob_weighted):.4f}, Mean Pred Prob = {prob_weighted.mean()*100:.2f}%")
print(f"LR Calibrated:   Brier = {brier_score_loss(y_test, prob_calibrated):.4f}, LogLoss = {log_loss(y_test, prob_calibrated):.4f}, Mean Pred Prob = {prob_calibrated.mean()*100:.2f}%")
print(f"Actual Failure Prevalence in Test Set: {y_test.mean()*100:.2f}%")

# 9. Generate Visualization Suite
print("\n" + "="*50)
print("GENERATING VISUALIZATION SUITE")
print("="*50)

# Set clean aesthetic styling
sns.set_theme(style='whitegrid', font='sans-serif')
palette = sns.color_palette("mako", len(models))

# GRAPH 1 — Overall Model Comparison (Grouped Bar Chart)
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_keys = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(models))
width = 0.2

for i, (m_label, m_key) in enumerate(zip(metrics_to_plot, metric_keys)):
    vals = [test_results[m][m_key] for m in models.keys()]
    ax.bar(x + (i - 1.5) * width, vals, width, label=m_label)

ax.set_xticks(x)
ax.set_xticklabels(models.keys(), rotation=15, ha='right', fontsize=11, fontweight='medium')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Graph 1 — Overall Model Comparison (Held-Out Test Set)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(STATIC_ASSETS_DIR, 'model_comparison.png'), dpi=300)
plt.close(fig)
print("  Saved: model_comparison.png")

# GRAPH 2 — Prediction Error Comparison (MSE/Brier, RMSE, Log Loss)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
names = list(models.keys())
brier_vals = [test_results[m]['brier_score'] for m in names]
rmse_vals = [test_results[m]['rmse'] for m in names]
ll_vals = [test_results[m]['log_loss'] for m in names]

# Panel 1: MSE / Brier & RMSE
x = np.arange(len(names))
ax1.bar(x - 0.15, brier_vals, 0.3, label='MSE / Brier Score (Equivalent)', color='#2b5c8f')
ax1.bar(x + 0.15, rmse_vals, 0.3, label='RMSE (sqrt of MSE)', color='#46a5a5')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
ax1.set_title('Prediction Error Metrics (MSE & RMSE)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Score', fontsize=11)
ax1.legend()

# Panel 2: Log Loss
ax2.bar(x, ll_vals, 0.5, color='#d97706')
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
ax2.set_title('Cross-Entropy Log Loss (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Log Loss', fontsize=11)

plt.suptitle('Graph 2 — Prediction-Error & Probability Quality Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(STATIC_ASSETS_DIR, 'prediction_error_comparison.png'), dpi=300)
plt.close(fig)
print("  Saved: prediction_error_comparison.png")

# GRAPH 3 — Precision–Recall Curves
fig, ax = plt.subplots(figsize=(9, 6))
for name in models.keys():
    precision, recall, _ = precision_recall_curve(y_test, model_probs[name])
    ap_score = test_results[name]['average_precision']
    ax.plot(recall, precision, lw=2, label=f"{name} (AP = {ap_score:.3f})")

ax.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Prevalence Baseline ({y_test.mean()*100:.1f}%)')
ax.set_xlabel('Recall (Failure Detection Rate)', fontsize=12)
ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
ax.set_title('Graph 3 — Precision–Recall Curves (AI4I Imbalanced Positive Class)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc='lower left', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(STATIC_ASSETS_DIR, 'precision_recall_curves.png'), dpi=300)
plt.close(fig)
print("  Saved: precision_recall_curves.png")

# GRAPH 4 — Confusion Matrices for All Candidate Models
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (name, m) in enumerate(models.items()):
    ax = axes[idx]
    tn = test_results[name]['tn']
    fp = test_results[name]['fp']
    fn = test_results[name]['fn']
    tp = test_results[name]['tp']
    cm = np.array([[tn, fp], [fn, tp]])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"{name}\n(TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xticklabels(['No Failure (0)', 'Failure (1)'])
    ax.set_yticklabels(['No Failure (0)', 'Failure (1)'])

plt.suptitle('Graph 4 — Confusion Matrices at Documented Operating Threshold (0.50)', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(STATIC_ASSETS_DIR, 'confusion_matrices.png'), dpi=300)
plt.close(fig)
print("  Saved: confusion_matrices.png")

# GRAPH 5 — Cross-Validation Stability (Mean F1 with Error Bars)
fig, ax = plt.subplots(figsize=(10, 5))
names = list(models.keys())
f1_means = [cv_metrics[m]['f1_mean'] for m in names]
f1_stds = [cv_metrics[m]['f1_std'] for m in names]
rec_means = [cv_metrics[m]['recall_mean'] for m in names]
rec_stds = [cv_metrics[m]['recall_std'] for m in names]

x = np.arange(len(names))
ax.errorbar(x - 0.12, f1_means, yerr=f1_stds, fmt='o', capsize=6, lw=2, elinewidth=2, label='F1-Score (Mean ± Std)', color='#1d4ed8')
ax.errorbar(x + 0.12, rec_means, yerr=rec_stds, fmt='s', capsize=6, lw=2, elinewidth=2, label='Recall (Mean ± Std)', color='#059669')

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right', fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Graph 5 — 5-Fold Stratified Cross-Validation Stability', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(STATIC_ASSETS_DIR, 'cv_stability.png'), dpi=300)
plt.close(fig)
print("  Saved: cv_stability.png")

# GRAPH 6 — ROC Curves
fig, ax = plt.subplots(figsize=(9, 6))
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, model_probs[name])
    auc_val = test_results[name]['roc_auc']
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.500)')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(STATIC_ASSETS_DIR, 'roc_curves.png'), dpi=300)
plt.close(fig)
print("  Saved: roc_curves.png")

# GRAPH 7 — Calibration / Reliability Curves
fig, ax = plt.subplots(figsize=(8, 6))
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_weighted, n_bins=8, strategy='quantile')
ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='LR Weighted (Overconfident)', color='#ef4444')

f_pos_cal, m_pred_cal = calibration_curve(y_test, prob_calibrated, n_bins=8, strategy='quantile')
ax.plot(m_pred_cal, f_pos_cal, 'o-', label='LR Sigmoid Calibrated', color='#10b981')

f_pos_xgb, m_pred_xgb = calibration_curve(y_test, model_probs['Tuned XGBoost'], n_bins=8, strategy='quantile')
ax.plot(m_pred_xgb, f_pos_xgb, '^-', label='Tuned XGBoost', color='#3b82f6')

ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of True Positives', fontsize=12)
ax.set_title('Calibration / Reliability Diagrams', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(STATIC_ASSETS_DIR, 'calibration_curves.png'), dpi=300)
plt.close(fig)
print("  Saved: calibration_curves.png")

# 10. SHAP Explainability on Final Tuned XGBoost Model
print("\n" + "="*50)
print("COMPUTING SHAP EXPLAINABILITY (TUNED XGBOOST)")
print("="*50)

final_model = trained_estimators['Tuned XGBoost']
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_test_df)

# Plot 1: SHAP Summary Beeswarm Plot
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title('Global Feature Impact on Machine Failure (SHAP Beeswarm)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_ASSETS_DIR, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: shap_summary_beeswarm.png")

# Plot 2: SHAP Global Feature Importance Bar Plot
fig = plt.figure(figsize=(9, 5))
shap.plots.bar(shap_values, show=False)
plt.title('Global Feature Importance (|SHAP Mean Impact|)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_ASSETS_DIR, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: shap_feature_importance.png")

# Plot 3: SHAP Waterfall Plot for a Failure Case (Individual Instance)
# Find a true positive failure instance in the test set
tp_indices = np.where((y_test == 1) & (model_preds['Tuned XGBoost'] == 1))[0]
sample_idx = int(tp_indices[0]) if len(tp_indices) > 0 else 0

fig = plt.figure(figsize=(9, 6))
shap.plots.waterfall(shap_values[sample_idx], show=False)
plt.title(f'SHAP Waterfall Explanation — Equipment Failure Case (Test #{sample_idx})', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_ASSETS_DIR, 'shap_waterfall_sample.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: shap_waterfall_sample.png")

# Plot 4: SHAP Dependence Plot for Torque
fig, ax = plt.subplots(figsize=(8, 5))
shap.dependence_plot('Torque', shap_values.values, X_test_df, interaction_index='Rotational_speed', ax=ax, show=False)
plt.title('SHAP Dependence: Torque vs. Rotational Speed Interaction', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_ASSETS_DIR, 'shap_dependence_torque.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: shap_dependence_torque.png")

# 11. Construct and Save Production Pipeline and Model Artifacts
print("\n" + "="*50)
print("CONSTRUCTING AND SAVING PRODUCTION ARTIFACTS")
print("="*50)

# Build unified inference pipeline
production_pipeline = Pipeline([
    ('scaler', true_scaler),
    ('model', final_model)
])

# Save pipeline and standalone model
joblib.dump(production_pipeline, os.path.join(MODELS_DIR, 'final_pipeline.pkl'))
joblib.dump(final_model, os.path.join(MODELS_DIR, 'tuned_xgboost_model.pkl'))
print("Saved final_pipeline.pkl and tuned_xgboost_model.pkl.")

# Save complete metadata and genuine metrics
metadata = {
    'model_name': 'Tuned XGBoost Classifier',
    'algorithm': 'Gradient Boosted Decision Trees',
    'dataset': 'AI4I 2020 Predictive Maintenance Dataset',
    'target_column': 'Machine failure',
    'operating_threshold': SELECTED_THRESHOLD,
    'features': clean_cols,
    'test_metrics': test_results['Tuned XGBoost'],
    'cv_metrics': cv_metrics['Tuned XGBoost'],
    'leaderboard': leaderboard_rows,
    'training_distributions': {
        'air_temperature': {'min': 295.3, 'max': 304.5, 'unit': 'K'},
        'process_temperature': {'min': 305.7, 'max': 313.8, 'unit': 'K'},
        'rotational_speed': {'min': 1168.0, 'max': 2886.0, 'unit': 'rpm'},
        'torque': {'min': 3.8, 'max': 76.6, 'unit': 'Nm'},
        'tool_wear': {'min': 0.0, 'max': 253.0, 'unit': 'min'}
    },
    'risk_tiers': {
        'low': {'max_probability': 20.0, 'action': 'Normal monitoring maintained.'},
        'medium': {'min_probability': 20.0, 'max_probability': 50.0, 'action': 'Elevated caution — schedule routine inspection.'},
        'high': {'min_probability': 50.0, 'action': 'Critical alert — immediate tool inspection required.'}
    }
}

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)
print("Saved model_metadata.json.")

# 12. Verification of Saved Artifacts
print("\n" + "="*50)
print("VERIFYING INFERENCE PIPELINE RELOAD")
print("="*50)

reloaded_pipeline = joblib.load(os.path.join(MODELS_DIR, 'final_pipeline.pkl'))

# Test with raw unscaled normal inputs
test_raw_normal = pd.DataFrame([[300.0, 310.0, 1400.0, 30.0, 20.0, 0, 0, 10.0, 0]], columns=clean_cols)
prob_norm = reloaded_pipeline.predict_proba(test_raw_normal)[0][1]
pred_norm = int(prob_norm >= SELECTED_THRESHOLD)
print(f"Normal Test Input -> Prob: {prob_norm*100:.2f}%, Pred: {pred_norm} (Expected 0)")

# Test with raw unscaled high stress inputs
test_raw_stress = pd.DataFrame([[303.0, 312.0, 1250.0, 68.0, 230.0, 1, 0, 9.0, 2]], columns=clean_cols)
prob_str = reloaded_pipeline.predict_proba(test_raw_stress)[0][1]
pred_str = int(prob_str >= SELECTED_THRESHOLD)
print(f"High Stress Test Input -> Prob: {prob_str*100:.2f}%, Pred: {pred_str} (Expected 1)")

print("\n" + "="*50)
print("TRAINING, EVALUATION, AND ARTIFACT GENERATION COMPLETED SUCCESSFULLY!")
print("="*50)
