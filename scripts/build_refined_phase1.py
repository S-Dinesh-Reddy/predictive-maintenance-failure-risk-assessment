import json
import os

def make_cell(cell_type, source_text):
    lines = [line + '\n' for line in source_text.strip().split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
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

# Title & Overview
cells.append(make_cell("markdown", """# Phase 1: Data Ingestion, Data Cleaning, Exploratory Data Analysis & Feature Engineering

**Author:** Dinesh Reddy  
**Project:** AI4I 2020 Predictive Maintenance & Machine Failure Risk Assessment  
**Objective:** End-to-end data preparation, target leakage prevention, exploratory analysis, domain feature engineering, class imbalance diagnosis, and stratified data splitting for predictive modeling.

---

### Phase 1 Analytical Workflow
1. **Data Ingestion & Integrity Checks**: Verify row/column integrity, missing values, duplicates, and physical sensor types.
2. **Identifier & Leakage Removal**: Drop non-predictive identifiers (`UDI`, `Product ID`) and downstream failure-mode indicator flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to prevent target leakage into `Machine failure`.
3. **Categorical Encoding**: One-hot encode the machine product variant (`Type` $\\in \\{L, M, H\\}$) with `drop_first=True`.
4. **Class Imbalance & EDA**: Analyze the severe class imbalance (~3.39% failure prevalence), sensor distributions, and bivariate relationships with machine failure.
5. **Domain Feature Engineering**:
   - Construct `Temperature difference [K] = Process temperature [K] - Air temperature [K]`.
   - Discretize tool degradation into 3 quantiles: `Tool wear level` (0: Low, 1: Medium, 2: High).
6. **Stratified Partitioning & Feature Scaling**: Perform an 80/20 stratified train-test split (`random_state=42`) and fit `StandardScaler` strictly on the training set.
7. **Validation Infrastructure**: Calculate balanced class weights and verify Stratified 5-Fold Cross-Validation fold consistency."""))

# Cell 0: Imports
cells.append(make_cell("code", """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling for academic presentation
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True"""))

# Cell 1 & 2: Data Loading with robust local/Colab support
cells.append(make_cell("markdown", """## 1. Dataset Ingestion & Initial Integrity Inspection"""))

cells.append(make_cell("code", """# Resolve dataset path dynamically across local and Google Colab environments
DATA_DIR = os.path.join('..', 'data') if os.path.exists(os.path.join('..', 'data')) else 'data'
csv_path = os.path.join(DATA_DIR, 'ai4i2020.csv')

if not os.path.exists(csv_path):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        csv_path = '/content/drive/MyDrive/Predictive_Maintenance/ai4i2020.csv'
    except Exception:
        pass

df = pd.read_csv(csv_path)
print(f"Dataset successfully loaded from: {csv_path}")
print(f"Dataset Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")"""))

# Cell 3: head
cells.append(make_cell("code", """df.head()"""))

# Cell 4: shape
cells.append(make_cell("code", """df.shape"""))

# Cell 5: info
cells.append(make_cell("code", """df.info()"""))

# Cell 6: null count
cells.append(make_cell("code", """df.isnull().sum()"""))

# Cell 7: duplicated count
cells.append(make_cell("code", """df.duplicated().sum()"""))

# Cell 8: columns
cells.append(make_cell("code", """df.columns"""))

# Cell 9: Type value counts
cells.append(make_cell("code", """df['Type'].value_counts()"""))

# Cell 10: Product ID nunique
cells.append(make_cell("code", """df['Product ID'].nunique()"""))

# Cell 11: Machine failure counts
cells.append(make_cell("code", """df['Machine failure'].value_counts()"""))

# Cell 12: Failure modes sum
cells.append(make_cell("markdown", """## 2. Identifier Removal & Target Leakage Prevention

The AI4I 2020 dataset includes five binary failure-mode columns:
- `TWF`: Tool Wear Failure
- `HDF`: Heat Dissipation Failure
- `PWF`: Power Failure
- `OSF`: Overstrain Failure
- `RNF`: Random Failure

These flags are causal breakdowns of the target (`Machine failure`). Including them as predictors would cause catastrophic **target leakage** because they are only known after a failure has already occurred. Furthermore, `UDI` (record index) and `Product ID` (variant identifier) are non-predictive keys and must be removed."""))

cells.append(make_cell("code", """failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df[failure_modes].sum()"""))

# Cell 13: copy
cells.append(make_cell("code", """df_clean = df.copy()"""))

# Cell 14: drop identifiers and leakage columns
cells.append(make_cell("code", """df_clean = df_clean.drop(columns=[
    'UDI',
    'Product ID',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
])"""))

# Cell 15: columns after drop
cells.append(make_cell("code", """df_clean.columns"""))

# Cell 16: One-hot encode Type
cells.append(make_cell("markdown", """## 3. Categorical Encoding (Machine Variant)

The machine type represents quality variants: Low (L: 50%), Medium (M: 30%), High (H: 20%). We one-hot encode `Type` with `drop_first=True`, creating `Type_L` and `Type_M` as integer binary columns (Type H acts as the reference baseline)."""))

cells.append(make_cell("code", """df_clean = pd.get_dummies(
    df_clean,
    columns=['Type'],
    drop_first=True,
    dtype=int
)"""))

# Cell 17: head
cells.append(make_cell("code", """df_clean.head()"""))

# Cell 18: columns
cells.append(make_cell("code", """df_clean.columns"""))

# Cell 19: dtypes
cells.append(make_cell("code", """df_clean.dtypes"""))

# Cell 20: describe
cells.append(make_cell("code", """df_clean.describe()"""))

# Cell 21 & 22: Class distribution & visualization
cells.append(make_cell("markdown", """## 4. Class Imbalance Analysis

Machine failures account for approximately 3.39% of the dataset (339 failure records out of 10,000). Visualizing this severe skew demonstrates why standard classification accuracy is misleading and why cost-sensitive weighting and stratified evaluation are necessary."""))

cells.append(make_cell("code", """df_clean['Machine failure'].value_counts(normalize=True) * 100"""))

cells.append(make_cell("code", """counts = df_clean['Machine failure'].value_counts().sort_index()
labels = ['No Failure (0)', 'Machine Failure (1)']

plt.figure(figsize=(7, 4.5))
bars = plt.bar(labels, counts.values, color=['#2563eb', '#dc2626'], edgecolor='black', alpha=0.85)
plt.title('Machine Failure Class Distribution (AI4I 2020)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Machine State', fontsize=11)
plt.ylabel('Number of Observations', fontsize=11)
plt.ylim(0, 11000)

for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df_clean)) * 100
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 150, f"{yval:,} ({pct:.2f}%)", ha='center', va='bottom', fontweight='bold')

plt.show()"""))

# Cell 23: mean sensor values grouped by failure
cells.append(make_cell("markdown", """## 5. Exploratory Data Analysis: Sensor Distributions & Operating Regimes"""))

cells.append(make_cell("code", """df_clean.groupby('Machine failure')[[
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]].mean()"""))

# Cell 24: KDE-style numeric distributions
cells.append(make_cell("code", """features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

for feature in features:
    plt.figure(figsize=(7, 3.8))
    sns.kdeplot(
        data=df_clean[df_clean['Machine failure'] == 0],
        x=feature,
        fill=True,
        alpha=0.25,
        color='#2563eb',
        label='No Failure (0)',
        linewidth=2.0
    )
    sns.kdeplot(
        data=df_clean[df_clean['Machine failure'] == 1],
        x=feature,
        fill=True,
        alpha=0.35,
        color='#dc2626',
        label='Failure (1)',
        linewidth=2.2
    )
    plt.xlabel(feature, fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title(f'{feature}: KDE Distribution by Machine State', fontsize=12, fontweight='bold', pad=10)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()"""))

# Cell 25: boxplots
cells.append(make_cell("code", """for feature in features:
    plt.figure(figsize=(6, 3.8))
    df_clean.boxplot(
        column=feature,
        by='Machine failure',
        boxprops=dict(color='#2563eb'),
        medianprops=dict(color='#dc2626', linewidth=2)
    )
    plt.title(f'{feature}: Boxplot by Failure State', fontsize=12, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Machine Failure (0 = Normal, 1 = Failed)', fontsize=11)
    plt.ylabel(feature, fontsize=11)
    plt.show()"""))

# Cell 26: correlation heatmap
cells.append(make_cell("markdown", """## 6. Correlation Analysis

Thermal and rotational sensors exhibit distinct physical dependencies (e.g. process temperature strongly tracks air temperature; rotational speed and torque exhibit an inverse mechanical power curve)."""))

cells.append(make_cell("code", """correlation = df_clean.corr()

plt.figure(figsize=(9, 6.5))
sns.heatmap(
    correlation,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5
)
plt.title('Correlation Matrix of Cleaned Features', fontsize=13, fontweight='bold', pad=12)
plt.show()"""))

# Cell 27 & 28: Temperature difference
cells.append(make_cell("markdown", """## 7. Domain Feature Engineering

We engineer two physically motivated features:
1. **Temperature Difference**: $\\Delta T = \\text{Process temperature [K]} - \\text{Air temperature [K]}$. Heat dissipation failure occurs when heat generated during cutting cannot be transferred to the ambient environment.
2. **Tool Wear Level**: Continuous tool wear is discretized into 3 equal-frequency terciles (Low: 0, Medium: 1, High: 2) to capture threshold-based mechanical tool degradation."""))

cells.append(make_cell("code", """df_clean['Temperature difference [K]'] = (
    df_clean['Process temperature [K]']
    - df_clean['Air temperature [K]']
)"""))

cells.append(make_cell("code", """df_clean[['Air temperature [K]',
          'Process temperature [K]',
          'Temperature difference [K]']].head()"""))

# Cell 36: Tool wear continuous summary
cells.append(make_cell("code", """df_clean['Tool wear [min]'].describe()"""))

# Cell 37: Preprocessing Leakage Prevention
cells.append(make_cell("code", """# Preprocessing Leakage Prevention: Quantile boundaries for Tool wear level are
# fitted strictly on the training partition (80% stratified split) and subsequently applied.
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    df_clean.index,
    test_size=0.2,
    stratify=df_clean['Machine failure'],
    random_state=42
)

# Learn quantile boundaries strictly on training partition
_, tool_wear_bins = pd.qcut(
    df_clean.loc[train_idx, 'Tool wear [min]'],
    q=3,
    retbins=True,
    labels=['Low', 'Medium', 'High']
)
# Extend outer edges to -inf / +inf for complete operational domain coverage
tool_wear_bins_robust = [-np.inf, tool_wear_bins[1], tool_wear_bins[2], np.inf]
print(f"Training-Derived Tercile Bins: {tool_wear_bins}")
print(f"Operational Cutoffs: Low <= {tool_wear_bins[1]:.0f} min, Medium <= {tool_wear_bins[2]:.0f} min, High > {tool_wear_bins[2]:.0f} min")

# Apply transformation
df_clean['Tool wear level'] = pd.cut(
    df_clean['Tool wear [min]'],
    bins=tool_wear_bins_robust,
    labels=['Low', 'Medium', 'High']
)"""))

cells.append(make_cell("code", """df_clean['Tool wear level'].value_counts()"""))

cells.append(make_cell("code", """df_clean[['Tool wear [min]', 'Tool wear level']].head(10)"""))

cells.append(make_cell("code", """df_clean.groupby('Tool wear level')['Machine failure'].agg(
    ['count', 'sum', 'mean']
)"""))

cells.append(make_cell("code", """df_clean['Tool wear level'] = df_clean['Tool wear level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})"""))

cells.append(make_cell("code", """df_clean['Tool wear level'].value_counts().sort_index()"""))

cells.append(make_cell("code", """df_clean.head()"""))

cells.append(make_cell("code", """df_clean.columns"""))

cells.append(make_cell("code", """df_clean.groupby(['Type_L', 'Type_M'])['Machine failure'].agg(
    ['count', 'sum', 'mean']
)"""))

cells.append(make_cell("code", """df_clean['Machine failure'].value_counts()"""))

# Cell 40 & 41: Separate X and y
cells.append(make_cell("markdown", """## 8. Stratified Train-Test Splitting (80/20)

We isolate the feature matrix $X$ and binary label $y$. We then apply an exact **80/20 stratified split** with `random_state=42` to guarantee that the 3.39% failure rate is identically preserved in both subsets without data leakage."""))

cells.append(make_cell("code", """X = df_clean.drop(columns=['Machine failure'])
y = df_clean['Machine failure']"""))

cells.append(make_cell("code", """print("Features shape:", X.shape)
print("Target shape:", y.shape)"""))

# Cell 42: train_test_split
cells.append(make_cell("code", """from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)"""))

cells.append(make_cell("code", """print("Training set:", y_train.value_counts(normalize=True) * 100)
print("Test set:", y_test.value_counts(normalize=True) * 100)"""))

# Cell 44 & 45: StandardScaler
cells.append(make_cell("markdown", """## 9. Feature Scaling via StandardScaler

To avoid data leakage, `StandardScaler` is fitted **strictly on `X_train`** ($N=8,000$). The fitted transformation is then applied to transform `X_train` and `X_test`."""))

cells.append(make_cell("code", """from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)"""))

cells.append(make_cell("code", """print("Training shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)"""))

# Cell 46: compute_class_weight
cells.append(make_cell("markdown", """## 10. Class Weights & Stratified 5-Fold Validation Setup"""))

cells.append(make_cell("code", """from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)"""))

# Cell 47 & 48: StratifiedKFold
cells.append(make_cell("code", """from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Number of folds:", skf.get_n_splits())"""))

cells.append(make_cell("code", """for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
    print(
        f"Fold {fold}: "
        f"Train failure rate = {y_train.iloc[train_idx].mean():.4f}, "
        f"Validation failure rate = {y_train.iloc[val_idx].mean():.4f}"
    )"""))

# Cell 49: Final check
cells.append(make_cell("code", """# Final check before model training

print("Final feature shape:", X_train_scaled.shape)
print("Final target shape:", y_train.shape)

print("\\nTraining data:")
print(X_train_scaled.shape)

print("\\nTest data:")
print(X_test_scaled.shape)

print("\\nTarget distribution:")
print(y_train.value_counts())"""))

# Cell 50: Save datasets
cells.append(make_cell("markdown", """## 11. Processed Dataset Persistence

The processed and stratified feature matrices and target vectors are exported for Phase 2 model training."""))

cells.append(make_cell("code", """# Save processed datasets for the next ML phase
DATA_DIR = os.path.join('..', 'data') if os.path.exists(os.path.join('..', 'data')) else 'data'
os.makedirs(DATA_DIR, exist_ok=True)

pd.DataFrame(
    X_train_scaled,
    columns=X.columns
).to_csv(
    os.path.join(DATA_DIR, 'X_train_scaled.csv'),
    index=False
)

pd.DataFrame(
    X_test_scaled,
    columns=X.columns
).to_csv(
    os.path.join(DATA_DIR, 'X_test_scaled.csv'),
    index=False
)

y_train.to_csv(
    os.path.join(DATA_DIR, 'y_train.csv'),
    index=False
)

y_test.to_csv(
    os.path.join(DATA_DIR, 'y_test.csv'),
    index=False
)

MODELS_DIR = os.path.join('..', 'models') if os.path.exists(os.path.join('..', 'models')) else 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
import joblib
scaler.feature_names_in_ = np.array(X.columns)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

print(f"Processed datasets saved successfully to {os.path.abspath(DATA_DIR)}.")
print(f"Fitted StandardScaler saved successfully to {os.path.abspath(os.path.join(MODELS_DIR, 'scaler.pkl'))}.")"""))

# Summary Markdown
cells.append(make_cell("markdown", """---

# Phase 1 Complete — Data Preprocessing

The AI4I 2020 dataset was loaded, inspected, cleaned, transformed, and prepared for machine learning.

### Completed Tasks
- **Dataset Loading & Inspection**: Verified 10,000 observations across 14 raw columns; confirmed 0 missing values and 0 duplicate rows.
- **Identifier & Leakage Prevention**: Dropped non-predictive `UDI` and `Product ID`, as well as downstream failure-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
- **Categorical Feature Encoding**: One-hot encoded `Type` into `Type_L` and `Type_M`.
- **Domain Feature Engineering**: Formulated thermal delta `Temperature difference [K]` and quantized `Tool wear level`.
- **Exploratory Data Analysis**: Visualized class imbalance (3.39% failure rate), univariate sensor distributions, boxplots, and correlation heatmap.
- **Stratified Partitioning**: Executed 80/20 train/test split preserving exact class prevalence.
- **Feature Scaling**: Fitted `StandardScaler` strictly on training data ($N=8,000$).
- **Validation Infrastructure**: Computed inverse class weights and verified fold consistency across 5 stratified folds.

### Output Dataset Specifications
- Training Samples: 8,000 ($N_{\\text{normal}} = 7,729$, $N_{\\text{failure}} = 271$)
- Test Samples: 2,000 ($N_{\\text{normal}} = 1,932$, $N_{\\text{failure}} = 68$)
- Engineered Features: 9
- Target: `Machine failure` $\\in \\{0, 1\\}$
- All artifacts exported cleanly to `data/` for Phase 2 training."""))

notebook_data = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11.16"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

target_path = os.path.join('notebooks', 'Phase_1_Data_Preprocessing.ipynb')
with open(target_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=1)

print(f"Refined Phase 1 notebook written to {target_path} with {len(cells)} cells.")
