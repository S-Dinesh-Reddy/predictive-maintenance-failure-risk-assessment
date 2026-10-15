# Project Notebooks

This directory contains the core machine learning notebooks for the **AI4I 2020 Predictive Maintenance & Machine Failure Risk Assessment** project.

---

## 📘 Notebook Overview

### 1. `Phase_1_Data_Preprocessing.ipynb`
- **Cell Count**: 63 cells (50 code cells executed, 13 structured markdown cells, 0 errors).
- **Execution**: Top-to-bottom executable from a fresh kernel via Jupyter `nbconvert`.
- **Workflow**:
  - Ingestion of raw dataset (`data/ai4i2020.csv` / Colab fallback) with 10,000 samples and 14 raw columns.
  - Data integrity verification: 0 missing values, 0 duplicate records.
  - Target leakage prevention: strict exclusion of machine failure causes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and non-predictive identifiers (`UDI`, `Product ID`).
  - Categorical encoding: Machine `Type` one-hot encoded with reference category `Type_H`, yielding `Type_L` and `Type_M`.
  - Class imbalance diagnostics: 3.39% empirical failure rate (339 failure records out of 10,000).
  - Exploratory data analysis (EDA): Univariate sensor histograms, boxplots by machine state, and correlation heatmap.
  - Domain feature engineering:
    - Temperature Difference: $\Delta T = \text{Process temperature [K]} - \text{Air temperature [K]}$
    - Tool Wear Level: Quantile-binned tool life stage (0 = Low, 1 = Medium, 2 = High).
  - Stratified 80/20 train/test split with deterministic random seed (`random_state=42`).
  - Standard scaling fitted strictly on training data ($N=8,000$).
  - Inverse balanced class-weight computation (`scale_pos_weight = 28.52`).
  - Stratified 5-Fold validation check.
  - Export of processed datasets to `data/` (`X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`).

### 2. `Phase_2_Model_Training_Evaluation.ipynb`
- **Cell Count**: 112 cells (86 code cells executed, 26 structured markdown cells, 0 errors).
- **Execution**: Top-to-bottom executable from a fresh kernel via Jupyter `nbconvert`.
- **Workflow**:
  - Preprocessed dataset ingestion from `data/`.
  - Class distribution and 2D PCA feature-space projection.
  - Granular evaluation across five baseline algorithms:
    - Logistic Regression (Balanced linear log-odds baseline)
    - Random Forest Classifier (200 trees, max_depth=10, balanced)
    - Support Vector Machine (RBF kernel, 5-fold calibrated probabilities)
    - Decision Tree Classifier (max_depth=6, hierarchical plot_tree visualization)
    - Baseline XGBoost Classifier (`scale_pos_weight=28.52`)
  - Initial baseline model comparison table and grouped performance bar chart.
  - Multi-model ROC curves and Precision–Recall curves with class-prevalence baseline.
  - Hyperparameter optimization: `RandomizedSearchCV` across 30 iterations on 6 hyperparameters using 5-fold cross-validation on training data.
  - **Training-Only Model Selection**: Candidate architectures ranked strictly on 5-fold cross-validation results prior to test set evaluation. Tuned XGBoost selected as the winning architecture (CV F1 = 0.7783 ± 0.0276).
  - **Training-Only Threshold Optimization**: Operating threshold sweep ($[0.10, 0.90]$) conducted on Out-Of-Fold (OOF) probability predictions generated specifically from the tuned model configuration. Selected threshold = 0.50 based on operational maintenance cost trade-offs.
  - **Held-Out Test Set Evaluation**: Unbiased benchmarking on the unobserved 2,000-sample test set at the 0.50 threshold.
  - **Advanced Diagnostics & Error Analysis**:
    - Mathematical equivalence of MSE and Brier score for binary classifications.
    - Probability calibration analysis: why low Brier score alone does not guarantee calibration in rare-event domains (~3.4%).
    - Cross-validation fold stability error bars.
    - 2x3 confusion matrix grid across all candidate models.
  - **Explainable AI (SHAP)**:
    - Global Beeswarm summary plot.
    - Mean absolute feature importance ranking.
    - Local waterfall case study explaining a true equipment failure event.
    - Spindle torque vs. rotational speed bivariate dependence interaction plot.
  - **Production Pipeline & Scaler Verification**:
    - Reconstructed authoritative `StandardScaler` with proven Phase 1 training set statistics.
    - End-to-end `Pipeline` assembly, smoke test, artifact export (`models/`), and reload parity verification.
  - Baseline vs. Tuned comparison chart and final gain-based feature importance plot.
  - Comprehensive academic conclusion and governance summary.

---

## 🏃 Execution Commands

To execute the notebooks from a fresh kernel using the project virtual environment:

```powershell
# Execute Phase 1 Data Preprocessing
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/Phase_1_Data_Preprocessing.ipynb

# Execute Phase 2 Model Training & Evaluation
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/Phase_2_Model_Training_Evaluation.ipynb
```
