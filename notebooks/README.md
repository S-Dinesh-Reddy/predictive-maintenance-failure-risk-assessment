# Project Notebooks

This directory contains the authoritative, top-to-bottom executable Jupyter notebooks for the **Predictive Maintenance and Failure Risk Assessment for Industrial Equipment** project, utilizing the AI4I 2020 Predictive Maintenance Dataset.

---

## Notebook Overview & Execution Audit

### 1. `Phase_1_Data_Preprocessing.ipynb`
- **Cell Count:** **66 cells** (52 executed code cells, 14 structured markdown cells, 0 errors).
- **Execution Mode:** Completely reproducible and top-to-bottom executable from a clean kernel via Jupyter `nbconvert`.
- **Core Workflow & Methodological Milestones:**
  1. **Dataset Ingestion & Integrity Verification:** Ingests raw telemetry (`data/ai4i2020.csv` / Colab fallback) containing exactly 10,000 records and 14 raw columns. Confirms 0 missing values and 0 duplicate rows.
  2. **Target Leakage Elimination:** Strictly removes all five deterministic failure-cause indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and arbitrary metadata columns (`UDI`, `Product ID`) prior to feature extraction.
  3. **Categorical Feature Encoding:** One-hot encodes the machine `Type` variant using `Type_H` as the reference baseline, producing binary features `Type_L` (50% prevalence) and `Type_M` (30% prevalence).
  4. **Class Imbalance Quantification:** Documents the severe 28.5:1 imbalance in the target variable `Machine failure` (9,661 normal cycles vs. 339 failure cycles; 3.39% failure rate).
  5. **Exploratory Data Analysis (EDA):** Generates univariate histogram and KDE distributions for all continuous sensor features (Figures 2–6), boxplots stratified by machine failure state (Figures 7–11), bivariate scatter plots revealing empirical operating regimes and failure clusters (Figures 12–13), and a Pearson correlation heatmap (Figure 14).
  6. **Domain Feature Engineering:**
     - **Temperature Difference:** $\Delta T = \text{Process temperature [K]} - \text{Air temperature [K]}$, isolating localized cutting-zone thermal stress from ambient temperature variations.
     - **Tool Wear Level:** Quantile-binned ordinal tool life stage derived strictly from training data quantiles (0 = Low [0–71 min], 1 = Medium [72–144 min], 2 = High [145–253 min]).
  7. **Stratified 80/20 Partitioning:** Splits data into an 8,000-sample training set and a 2,000-sample held-out test set using `random_state=42` and `stratify=y`. The test set is sealed immediately.
  8. **Leakage-Free Standard Scaling:** Fits a `StandardScaler` strictly on the 8,000-sample training partition. The fitted scaler statistics (`mean_`, `scale_`) are documented for production provenance.
  9. **Inverse Class Weight Computation:** Computes exact analytical class weighting: $\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{7729}{271} = 28.5203$.
  10. **Dataset Export:** Serializes processed partitions to `data/` (`X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`).

---

### 2. `Phase_2_Model_Training_Evaluation.ipynb`
- **Cell Count:** **121 cells** (89 executed code cells, 32 structured markdown cells, 0 errors).
- **Execution Mode:** Completely reproducible and top-to-bottom executable from a clean kernel via Jupyter `nbconvert`.
- **Core Workflow & Methodological Milestones:**
  1. **Dataset Ingestion & Geometric Inspection:** Loads scaled training and test datasets. Projects 9-dimensional feature space onto 2D PCA space, illustrating non-linear failure boundary overlap.
  2. **Multi-Algorithm Baseline Benchmarking (Sections 4–11):** Trains and evaluates five diverse classification architectures on the 9 features using balanced weighting:
     - **Logistic Regression:** Linear log-odds baseline (Recall: 86.8%, Precision: 15.5%, 322 false alarms).
     - **Random Forest:** 200 trees, `max_depth=10`, balanced weights (F1: 72.2%, Recall: 83.8%, Precision: 63.3%, 33 false alarms).
     - **Support Vector Machine (RBF Kernel):** $C=1.0$, `gamma='scale'`, balanced weights, calibrated probabilities (Precision: 75.0%, Recall: 44.1%, 38 missed failures).
     - **Decision Tree Classifier:** `max_depth=6`, balanced weights, full `plot_tree` structural visualization (F1: 51.2%, Recall: 80.9%, Precision: 37.4%, 92 false alarms).
     - **Baseline XGBoost Classifier:** Default parameters with `scale_pos_weight=28.52` (F1: 66.3%, Recall: 85.3%, Precision: 54.2%, 49 false alarms).
  3. **Baseline Diagnostic Synthesis (Section 11):** Multi-model ROC curves (AUC: RF=0.979, XGB=0.976, SVM=0.968, LR=0.914, DT=0.843) and Precision–Recall curves against empirical class prevalence (3.39%).
  4. **Hyperparameter Optimization via `RandomizedSearchCV` (Section 12):** 30 candidate parameter combinations evaluated across Stratified 5-Fold Cross-Validation on the 8,000-sample training partition, optimizing for F1-score. Winning configuration achieves CV F1 = **0.7783 +/- 0.0276** (`n_estimators=269`, `max_depth=6`, `learning_rate=0.142009`, `min_child_weight=2`, `subsample=0.942436`, `colsample_bytree=0.879660`, `scale_pos_weight=28.5203`).
  5. **Training-Only Cross-Validation Model Selection:** Candidate models formally ranked on 5-fold training CV F1 before test evaluation:
     - Tuned XGBoost (0.7783) > Baseline XGBoost (0.6931) > Random Forest (0.6876) > Decision Tree (0.5027) > SVM (0.4985) > Logistic Regression (0.2433).
  6. **Stage-Wise Boosting Convergence Diagnostics across 269 Rounds (Sections 12.1 & 12.2):**
     - **Log Loss Learning Curve (Cell 84):** Evaluated on internal 80/20 train partition ($N_{\text{sub\_train}}=6,400$, $N_{\text{val}}=1,600$). Sub-training loss drops from $0.5856 \rightarrow 0.0029$. Internal validation loss reaches global minimum of **0.0586 at Round 114**, rising mildly to **0.0652 at Round 269** (mild late-round overfitting, bounded by subsampling regularization without divergence).
     - **Classification Metric Learning Curves (Cell 87):** Evaluates Accuracy (zoomed: 0.90–1.00), Precision, Recall, and F1 at fixed threshold $\theta = 0.50$. Initial Val F1 = 0.3962; peak Val F1 = **0.7379 at Round 168** (Val Acc: 98.31%, Val Prec: 77.55%, Val Rec: 70.37%); softening slightly to 0.7071 at Round 269 while sub-training reaches 0.9977.
     - **Faculty Review Justifications:** Explains sequential pseudo-residual boosting ($f_M(x) = \sum f_m(x)$) vs. neural network backpropagation epochs, rationale for internal training holdout, and why early stopping on a single holdout was not applied post-hoc to alter the 5-fold CV-selected configuration.
  7. **Training-Only Threshold Optimization (Section 13):** Evaluates Out-of-Fold (OOF) training probabilities across thresholds $[0.10, 0.90]$. Peak training OOF F1 occurs at $\theta = 0.50$ (0.7766), justifying retention of the 0.50 operational decision boundary.
  8. **Unbiased Held-Out Test Set Evaluation (Section 14):** Evaluates Tuned XGBoost on the untouched 2,000-sample test set (68 failures):
     - Accuracy: **98.65%** | Precision: **78.87%** | Recall: **82.35%** | F1-Score: **80.58%** | ROC-AUC: **97.58%** | Average Precision (AP): **86.01%**
     - Brier Score (MSE): **0.0117** | RMSE: **0.1081** | Log Loss: **0.0531** | Specificity: **99.22%** | MCC: **0.7990**
     - Confusion Matrix: $\text{TN} = 1917, \text{FP} = 15, \text{FN} = 12, \text{TP} = 56$.
     - Tuning reduced false alarms by **69.4%** (FP dropped from 49 to 15 vs. Baseline XGBoost).
  9. **Advanced Comparative Diagnostics & Error Analysis (Section 15):**
     - Multi-model prediction error bar chart (Brier score, RMSE, Log Loss).
     - Stratified 5-Fold cross-validation stability chart with error bars.
     - Master benchmark comparison table and grouped bar charts.
     - In-depth justification of why $R^2$ is mathematically inappropriate for binary classification under rare event conditions and why strictly proper scoring rules (Brier score, Log Loss) are mandatory.
  10. **Explainable AI (SHAP) with `shap.TreeExplainer` (Section 16):**
      - Global beeswarm summary plot across all test instances.
      - Mean absolute SHAP feature importance ranking:
        `Torque (31.48%)` > `Tool_wear (24.40%)` > `Rotational_speed (14.94%)` > `Temperature_difference (9.11%)` > `Process_temperature (6.64%)` > `Air_temperature (6.53%)` > `Tool_wear_level (3.50%)` > `Type_L (2.15%)` > `Type_M (1.25%)`.
        Top 3 physical drivers combined account for **70.82%** of total model attribution.
      - Conceptual distinction between additive SHAP values (log-odds impact across test samples) and XGBoost Gain (in-tree split impurity reduction during training).
      - Local SHAP waterfall case study explaining a true equipment failure event.
      - Bivariate dependence interaction plot for Spindle Torque vs. Rotational Speed.
  11. **Production Pipeline Assembly & Provenance Verification (Section 17):**
      - Reconstructs authoritative `StandardScaler` asserting exact numerical match with Phase 1 training means (`true_mean`).
      - Assembles end-to-end `Pipeline([('scaler', scaler), ('model', best_xgb_model)])`.
      - Serializes production artifacts to `models/` (`final_pipeline.pkl`, `model_metadata.json`, `scaler.pkl`, `tuned_xgboost_model.pkl`).
      - Reloads serialized pipeline and verifies prediction parity on normal and high-risk telemetry.

---

## Headless Execution Commands

To execute either notebook from a clean kernel and verify end-to-end reproduction:

```bash
# Execute Phase 1 Notebook
python -m nbconvert --to notebook --execute --output Phase_1_Run.ipynb notebooks/Phase_1_Data_Preprocessing.ipynb

# Execute Phase 2 Notebook
python -m nbconvert --to notebook --execute --output Phase_2_Run.ipynb notebooks/Phase_2_Model_Training_Evaluation.ipynb
```

> **Note on Execution Safety:** Using `--output <filename>.ipynb` executes all cells from a clean kernel while preserving original executed outputs in the authoritative notebook files.
