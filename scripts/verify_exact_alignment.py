import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss, balanced_accuracy_score, matthews_corrcoef
)

clean_cols = [
    'Air_temperature', 'Process_temperature', 'Rotational_speed',
    'Torque', 'Tool_wear', 'Type_L', 'Type_M',
    'Temperature_difference', 'Tool_wear_level'
]
X_train = pd.read_csv('data/X_train_scaled.csv')
X_test = pd.read_csv('data/X_test_scaled.csv')
X_train.columns = clean_cols
X_test.columns = clean_cols
y_train = pd.read_csv('data/y_train.csv')['Machine failure'].values
y_test = pd.read_csv('data/y_test.csv')['Machine failure'].values
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
    'SVM (RBF)': CalibratedClassifierCV(SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=42), cv=5),
    'Baseline XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'),
    'Tuned XGBoost': XGBClassifier(n_estimators=269, learning_rate=0.14200913727754388, max_depth=6, min_child_weight=2, subsample=0.942436113869325, colsample_bytree=0.8796596399465607, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
}

scoring = {
    'f1': 'f1', 'recall': 'recall', 'precision': 'precision',
    'roc_auc': 'roc_auc', 'average_precision': 'average_precision',
    'accuracy': 'accuracy'
}

print("="*60)
print("VERIFYING MODEL SUITE ALIGNMENT")
print("="*60)

for name, m in models.items():
    res = cross_validate(m, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    m.fit(X_train, y_train)
    probs = m.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    spec = tn / (tn + fp)
    bal_acc = balanced_accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    ll = log_loss(y_test, probs)
    print(f"\nModel: {name}")
    print(f"  CV 5-Fold: F1={res['test_f1'].mean():.4f} +/- {res['test_f1'].std():.4f} | Rec={res['test_recall'].mean():.4f} | Prec={res['test_precision'].mean():.4f} | AUC={res['test_roc_auc'].mean():.4f} | AP={res['test_average_precision'].mean():.4f}")
    print(f"  Test: F1={f1_score(y_test, preds):.4f} | Rec={recall_score(y_test, preds):.4f} | Prec={precision_score(y_test, preds):.4f} | AUC={roc_auc_score(y_test, probs):.4f} | AP={average_precision_score(y_test, probs):.4f}")
    print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn} | Spec={spec:.4f} | BalAcc={bal_acc:.4f} | MCC={mcc:.4f} | Brier={brier:.4f}")
