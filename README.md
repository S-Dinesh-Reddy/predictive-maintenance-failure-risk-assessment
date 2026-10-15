# Machine Learning-Based Predictive Maintenance and Failure Risk Assessment

A complete, reproducible, end-to-end machine learning system for predicting machine failure risk from industrial equipment operating conditions using the **AI4I 2020 Predictive Maintenance Dataset**.

The project implements a leakage-safe pipeline across data preprocessing, 5-fold cross-validation, model comparison across 5 baseline architectures + tuned XGBoost, probability calibration diagnostics, training-only threshold optimization, SHAP explainability, and deployment via an industrial Flask web application.

---

## 📌 Project Architecture & Workflow

```text
AI4I 2020 Dataset (10,000 records)
       │
       ▼
Phase 1: Preprocessing & Leakage Elimination (notebooks/Phase_1_Data_Preprocessing.ipynb - 63 cells)
  • Remove identifier columns (UDI, Product ID)
  • Drop failure mode target-derived causes (TWF, HDF, PWF, OSF, RNF)
  • One-hot encode Machine Type (Type_L, Type_M, Reference: Type_H)
  • Engineer domain features: Temperature difference (ΔT) & Tool wear level
  • Stratified 80/20 train/test split (Seed 42)
  • Fit StandardScaler on training data only (8,000 samples)
  • Verify class weights & 5-fold cross-validation consistency
       │
       ▼
Phase 2: Model Training, Evaluation & Calibration (notebooks/Phase_2_Model_Training_Evaluation.ipynb - 112 cells)
  • 5-Fold Stratified Cross-Validation on training data across 5 baseline algorithms:
    Logistic Regression, Decision Tree, Random Forest, SVM (RBF), Baseline XGBoost
  • Hyperparameter tuning: RandomizedSearchCV (30 iterations, 5-fold CV)
  • Training-Only Model Selection: Winner selected prior to test evaluation based on CV F1
  • Training-Only Threshold Optimization: Out-Of-Fold (OOF) probabilities from Tuned XGBoost
  • Final unbiased evaluation on held-out test partition (2,000 samples at 0.50 threshold)
  • Probability calibration diagnostics & Brier score / Log Loss error analysis
  • SHAP explainability (Global beeswarm, importance ranking, local waterfall case study, dependence)
       │
       ▼
Phase 3: Production Export & Deployment
  • Export leak-free production inference pipeline (`models/final_pipeline.pkl`)
  • Verified true scaler (`models/scaler.pkl`) and tuned model (`models/tuned_xgboost_model.pkl`)
  • Production metadata (`models/model_metadata.json`)
  • Industrial Flask dashboard (`app.py`, `templates/index.html`, `static/`)
  • Real-time two-level validation, risk tiering, telemetry logging, and SHAP visual diagnostics
```

---

## 📊 Dataset & Feature Schema

- **Dataset**: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository / Synthetic Milling Telemetry)
- **Total Records**: 10,000 samples
- **Target Variable**: `Machine failure` (Binary: 0 = Normal, 1 = Failure)
- **Natural Class Distribution**:
  - Training Partition (8,000 samples): 7,729 Normal (96.61%), 271 Failures (3.39%)
  - Held-out Test Partition (2,000 samples): 1,932 Normal (96.60%), 68 Failures (3.40%)
  - Imbalance Ratio (`scale_pos_weight`): **28.52**

### Features Used by Models & Pipeline (9 Total):
1. `Air_temperature`: Air temperature around the equipment [K]
2. `Process_temperature`: Process operating temperature [K]
3. `Rotational_speed`: Spindle rotational speed [rpm]
4. `Torque`: Spindle torque generated [Nm]
5. `Tool_wear`: Continuous tool cutting contact duration [min]
6. `Type_L`: Low-quality variant indicator (One-hot)
7. `Type_M`: Medium-quality variant indicator (One-hot)
8. `Temperature_difference`: Engineered thermal gradient: `Process_temperature − Air_temperature` [K]
9. `Tool_wear_level`: Engineered tool lifecycle stage (0 = Low ≤71 min, 1 = Medium 72–144 min, 2 = High >144 min)

---

## 🏆 Model Evaluation & Final Leaderboard

All models were evaluated using Stratified 5-Fold Cross-Validation on the training partition ($N=8,000$), followed by unbiased evaluation on the untouched held-out test set ($N=2,000$, 68 failures) at the documented operational decision threshold (0.50).

### Final Benchmark Leaderboard:

| Model Architecture | Test F1-Score | Test Recall | Test Precision | Test ROC-AUC | Test AP | Brier Score | Log Loss | FN (Missed) | FP (Alarms) | CV F1 (Mean ± Std) | CV Recall (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost** (Selected) | **80.58%** | **82.35%** | **78.87%** | **97.58%** | **86.01%** | **0.0117** | **0.0531** | **12** | **15** | **77.83% ± 2.76%** | **77.14% ± 4.37%** |
| **Random Forest** | 72.15% | 83.82% | 63.33% | 97.90% | 83.93% | 0.0187 | 0.0655 | 11 | 33 | 68.76% ± 4.24% | 75.64% ± 2.18% |
| **Baseline XGBoost** | 66.29% | 85.29% | 54.21% | 97.63% | 84.50% | 0.0205 | 0.0883 | 10 | 49 | 69.31% ± 5.07% | 80.07% ± 5.56% |
| **SVM (RBF Kernel)** | 55.56% | 44.12% | 75.00% | 96.78% | 65.93% | 0.0184 | 0.0961 | 38 | 10 | 49.73% ± 9.29% | 37.30% ± 9.42% |
| **Decision Tree** | 51.16% | 80.88% | 37.41% | 84.32% | 71.51% | 0.0399 | 0.8288 | 13 | 92 | 50.27% ± 2.04% | 80.83% ± 7.22% |
| **Logistic Regression** | 26.28% | 86.76% | 15.49% | 91.39% | 40.48% | 0.1199 | 0.3774 | 9 | 322 | 24.33% ± 1.21% | 81.93% ± 6.11% |

*Note: For binary classification probabilities, $\text{MSE} \equiv \text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (y_i - p_i)^2 = 0.0117$ for Tuned XGBoost.*

### Methodological Insights & Decision Rationale:
1. **Model Selection Independence**:
   - Model selection was finalized **strictly on training-only 5-fold cross-validation** before unsealing the held-out test set.
   - Tuned XGBoost achieved the highest training cross-validation F1 score (**0.7783 ± 0.0276**) and the most balanced recall/precision trade-off.
2. **Threshold Optimization & Asymmetric Risk Trade-off (Zero Test Leakage)**:
   - Out-of-fold (OOF) probability predictions were generated specifically from the tuned model architecture on training folds.
   - An empirical sweep across $[0.10, 0.90]$ revealed that threshold 0.65 yields a slightly higher out-of-fold F1 (0.7656 vs. 0.7541 at 0.50).
   - However, shifting from 0.50 to 0.65 reduces Recall from 76.38% to 72.32% (false negatives rise from 64 to 75 missed failures).
   - In industrial machining operations, an undetected machine failure causes catastrophic cutter breakage, spindle destruction, and unplanned downtime ($C_{\text{FN}} \gg C_{\text{FP}}$).
   - Operational threshold **0.50** is therefore chosen to prioritize higher failure capture (76.38% OOF recall; 82.35% held-out test recall) while preserving 74.46% OOF precision (78.87% test precision), rather than strictly maximizing the mathematical F1 score.
3. **Probability Calibration Nuance**:
   - Low Brier score alone does not establish calibration in imbalanced domains (~3.4% failures), because predicting $\hat{p} = 0.034$ everywhere yields Brier $\approx 0.033$.
   - Combining Brier score with Log Loss (0.0531) and Platt sigmoid calibration curve analysis proves genuine probabilistic fidelity.
4. **Dataset Scope & Industrial Deployment Limitations**:
   - The AI4I 2020 dataset is a synthetic benchmark reflecting mathematical tool degradation and thermal equations, but does not capture real-world factory noise, ambient temperature swings, or machine vibration drift.
   - Predictions serve as decision support and require plant-specific recalibration before autonomous shutdown deployment.

---

## 🔍 Explainable AI (SHAP)

SHAP `TreeExplainer` was applied to the selected Tuned XGBoost model:
- **Global Beeswarm Summary**: High Torque, high Tool Wear, and extreme Spindle Speeds are the primary risk drivers.
- **Feature Importance Ranking**: `Torque` > `Tool_wear` > `Rotational_speed` > `Air_temperature` > `Process_temperature`.
- **Local Waterfall Analysis**: Breaks down positive/negative feature contributions for a true equipment failure event.
- **Dependence Interaction**: Highlights the non-linear physical relationship between spindle torque and rotational speed.
- **Faculty Governance Note**: SHAP values represent statistical feature attributions within the learned decision surface, not physical proof of causation.

---

## 💻 Industrial Web Application (Flask)

The Flask web application provides an operator-ready predictive maintenance control center:
- **Production Pipeline**: Encapsulates true feature scaling and inference in `models/final_pipeline.pkl`.
- **Level 1 Hard Sanity Validation**: Rejects physically impossible / negative sensor inputs with HTTP 400.
- **Level 2 Training Distribution Check**: Alerts operators if sensor readings extrapolate outside observed training ranges without breaking inference.
- **Dynamic Risk Categorization**:
  - `LOW RISK`: Failure Probability < 20% (Normal Status)
  - `MEDIUM RISK`: 20% ≤ Failure Probability < 50% (Elevated Caution)
  - `HIGH RISK`: Failure Probability ≥ 50% (Maintenance Required)
- **Programmatic API Endpoints**:
  - `GET /api/metadata`: Returns model specifications, operational threshold, and test metrics.
  - `POST /api/predict`: JSON prediction endpoint with full input validation.

---

## 🚀 Setup and Execution Instructions

### 1. Environment Setup
The project requires Python 3.11+.

```powershell
# Using uv (Recommended)
uv venv .venv --python 3.11
uv pip install flask scikit-learn pandas numpy xgboost joblib matplotlib seaborn shap nbformat ipykernel nbconvert

# Or using standard pip
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execute Jupyter Notebooks Top-to-Bottom
```powershell
# Execute Phase 1 (Data Preprocessing & EDA - 63 cells)
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/Phase_1_Data_Preprocessing.ipynb

# Execute Phase 2 (Model Training & Evaluation - 112 cells)
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/Phase_2_Model_Training_Evaluation.ipynb
```

### 3. Run the Automated Test Suite
```powershell
.\.venv\Scripts\python.exe tests/test_pipeline_and_app.py
```
Validates required artifacts, dataset integrity, scaler parameters, pipeline inference, and Flask UI/API endpoints.

### 4. Launch the Flask Web Dashboard
```powershell
.\.venv\Scripts\python.exe app.py
```
Open your browser at `http://127.0.0.1:5000/`.

---

## ⚠️ Known Limitations & Domain Caveats

1. **Synthetic Nature of AI4I 2020**: The AI4I 2020 dataset is a synthetic benchmark reflecting milling machine physics. High test performance on this benchmark does not guarantee identical performance on physical factory milling machines without domain-specific telemetry calibration.
2. **Extrapolation Constraints**: The model was trained within specific sensor ranges (e.g., Torque 3.8–76.6 Nm, Air Temperature 295.3–304.5 K). Predictions on inputs outside these distributions represent extrapolation; the UI displays an advisory warning in such conditions.
3. **Attribution vs. Causation**: SHAP values explain model attribution, not mechanical causation. Physical maintenance decisions should be cross-verified against machine specifications and physical inspection.