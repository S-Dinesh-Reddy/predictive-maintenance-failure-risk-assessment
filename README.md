# Predictive Maintenance and Failure Risk Assessment for Industrial Equipment

An end-to-end machine learning system for equipment health monitoring, failure risk classification, and decision support using sensor telemetry from the **AI4I 2020 Predictive Maintenance Dataset**.

This project implements a leak-free machine learning workflow, evaluates six classification architectures under Stratified 5-Fold Cross-Validation, tracks stage-wise boosting convergence across 269 boosting rounds, optimizes decision boundaries using Out-of-Fold training predictions, provides game-theoretic model explainability using SHAP, and deploys a production-ready Flask decision-support web application and REST API.

---

## Live Application & Repository Links

- **Live Web Application:** [https://predictive-maintenance-failure-risk.onrender.com](https://predictive-maintenance-failure-risk.onrender.com)
- **GitHub Repository:** [https://github.com/S-Dinesh-Reddy/predictive-maintenance-failure-risk-assessment](https://github.com/S-Dinesh-Reddy/predictive-maintenance-failure-risk-assessment)
- **REST Metadata API:** [https://predictive-maintenance-failure-risk.onrender.com/api/metadata](https://predictive-maintenance-failure-risk.onrender.com/api/metadata)

> **Note on Cloud Hosting:** Hosted on Render's managed cloud platform. Free-tier containers spin down after 15 minutes of inactivity; initial cold starts take approximately 30–50 seconds, after which requests execute with sub-second latency.

---

## Problem Statement & Objectives

Modern manufacturing facilities experience significant financial losses from unexpected CNC machine breakdowns. Traditional maintenance strategies rely either on **reactive maintenance** (running tools to catastrophic failure, risking workpiece and spindle damage) or **calendar-based preventative maintenance** (replacing tools prematurely, leading to unnecessary tool costs and scheduled downtime).

### Project Objectives:
1. **Accurate Failure Detection:** Classify impending equipment failures from real-time physical sensor streams under severe class imbalance (~3.4% failure rate).
2. **False Alarm Reduction:** Maximize detection recall while preserving high precision to prevent unwarranted and costly production stoppages.
3. **Rigorous Leakage Control:** Enforce strict separation between training and evaluation partitions, preventing feature, target, or threshold leakage.
4. **Explainable AI (XAI):** Provide maintenance engineers with transparent, feature-level attributions for every prediction using game-theoretic SHAP values.
5. **Operational Deployment:** Deliver an intuitive operator dashboard and a programmatic REST API with two-level input validation and risk stratification.

---

## System Architecture & Workflow

The end-to-end machine learning engineering workflow is structured into three phases:

```
+---------------------------------------------------------------------------------+
|                    PHASE 1: DATA PREPROCESSING & LEAKAGE CONTROL                |
|  - AI4I 2020 Dataset (10,000 records, 3.39% failure prevalence)                 |
|  - Strict target leakage exclusion: TWF, HDF, PWF, OSF, RNF, UDI, Product ID   |
|  - Categorical encoding: Type_L, Type_M (Type_H reference baseline)             |
|  - Domain Feature Engineering: Delta T = Process Temp - Air Temp, Tool Wear Lvl |
|  - Stratified 80/20 Split (N_train = 8,000; N_test = 2,000 held-out)           |
|  - StandardScaler fitted strictly on training data                              |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                 PHASE 2: MODEL TRAINING, TUNING & ERROR ANALYSIS                |
|  - Baseline Architectures: Logistic Regression, Random Forest, SVM (RBF),       |
|    Decision Tree, Baseline XGBoost                                              |
|  - Training-Only Stratified 5-Fold Cross-Validation Model Selection            |
|  - Hyperparameter Optimization via RandomizedSearchCV (30 iterations)          |
|  - Winning Model: Tuned XGBoost (CV F1 = 0.7783 +/- 0.0276)                    |
|  - Stage-Wise Boosting Diagnostics across 269 Boosting Rounds:                  |
|    * Log Loss Learning Curve (min 0.0586 at Round 114; bounded late drift)      |
|    * 4 Classification Metric Curves (Accuracy, Precision, Recall, F1)           |
|  - Training-Only Out-of-Fold (OOF) Threshold Optimization at theta = 0.50       |
|  - Unbiased Final Evaluation on Held-Out Test Set (N = 2,000, 68 failures)      |
|  - SHAP Explainability: Beeswarm, Feature Attribution, Waterfall, Dependence    |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                 PHASE 3: PIPELINE SERIALIZATION & FLASK DEPLOYMENT              |
|  - Production Pipeline Assembly (StandardScaler + Tuned XGBoost Model)         |
|  - Serialization to models/final_pipeline.pkl and models/model_metadata.json   |
|  - Two-Level Input Validation: Level 1 Hard Limits & Level 2 Training Warnings  |
|  - Tri-Tier Risk Classification: Low (<20%), Medium (20-50%), High (>=50%)      |
|  - Interactive Dashboard & Programmatic REST API (POST /api/predict)           |
|  - 6-Suite Automated Regression Test Suite (tests/test_pipeline_and_app.py)    |
+---------------------------------------------------------------------------------+
```

---

## Repository Directory Structure

```
predictive-maintenance-failure-risk-assessment/
├── .gitignore                     # Git exclusion rules (caches, virtualenvs, outputs)
├── .python-version                # Python runtime specification (3.11)
├── README.md                      # Authoritative project documentation
├── requirements.txt               # Production and deployment dependencies
├── requirements-dev.txt           # Testing and development dependencies (pytest)
├── app.py                         # Production Flask web application & REST API
│
├── data/                          # Benchmark datasets and partitioned features
│   ├── ai4i2020.csv               # Raw AI4I 2020 dataset (10,000 records)
│   ├── X_train_scaled.csv         # Scaled training feature matrix (8,000 rows)
│   ├── X_test_scaled.csv          # Scaled held-out test feature matrix (2,000 rows)
│   ├── y_train.csv                # Training target labels (8,000 rows)
│   └── y_test.csv                 # Test target labels (2,000 rows)
│
├── docs/                          # Authoritative technical documentation
│   ├── Predictive_Maintenance_Project_Report.pdf  # Comprehensive technical report (32 pages)
│   └── Deployment_Documentation.pdf               # Production cloud runbook (6 pages)
│
├── models/                        # Serialized production pipeline artifacts
│   ├── final_pipeline.pkl         # End-to-end Pipeline (StandardScaler + Tuned XGBoost)
│   ├── tuned_xgboost_model.pkl    # Final Tuned XGBoost model (269 trees)
│   ├── scaler.pkl                 # Authoritative StandardScaler (N=8,000 provenance)
│   └── model_metadata.json        # Benchmark metrics, hyperparameters, and schemas
│
├── notebooks/                     # Authoritative, executable Jupyter notebooks
│   ├── Phase_1_Data_Preprocessing.ipynb           # Preprocessing, leakage control & EDA (66 cells)
│   ├── Phase_2_Model_Training_Evaluation.ipynb    # Modeling, tuning, curves & SHAP (121 cells)
│   └── README.md                                  # Notebook execution guide and cell audits
│
├── scripts/                       # Automated execution and utility scripts
│   ├── train_and_evaluate.py      # Standalone training, evaluation, and asset generation
│   ├── download_dataset.py        # Dataset verification and download helper
│   └── ...                        # Verification utilities
│
├── static/                        # Frontend assets and benchmark visualization plots
│   ├── style.css                  # UI styling and responsive layouts
│   ├── script.js                  # Frontend client validation and interaction logic
│   └── assets/                    # Benchmark graphs (Graphs 1-8) and UI imagery
│       ├── model_comparison.png
│       ├── prediction_error_comparison.png
│       ├── precision_recall_curves.png
│       ├── roc_curves.png
│       ├── confusion_matrices.png
│       ├── cv_stability.png
│       ├── calibration_curves.png
│       ├── learning_curve.png
│       └── shap_*.png
│
├── templates/                     # Flask Jinja2 HTML templates
│   └── index.html                 # Interactive operator dashboard template
│
└── tests/                         # Automated test suite
    └── test_pipeline_and_app.py   # 6-suite end-to-end regression test suite
```

---

## Dataset & Feature Schema

- **Dataset Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).
- **Total Records:** 10,000 operational cycles.
- **Target Variable:** `Machine failure` (binary: 0 = Normal Operation, 1 = Machine Failure).
- **Class Imbalance:** 9,661 Normal cycles (96.61%) vs. 339 Failure cycles (3.39%) &mdash; an imbalance ratio of ~28.5:1.
- **Target Leakage Safeguard:** All five failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and arbitrary identifiers (`UDI`, `Product ID`) were strictly eliminated prior to modeling.

### Feature Definitions (9 Features)

| Feature Name | Type | Physical Unit | Description | Observed Training Range |
|---|---|---|---|---|
| `Air_temperature` | Continuous | Kelvin [K] | Ambient temperature surrounding machine | [295.3, 304.5] K |
| `Process_temperature` | Continuous | Kelvin [K] | Cutting zone operating temperature | [305.7, 313.8] K |
| `Rotational_speed` | Continuous | rpm | CNC spindle motor rotational speed | [1,168, 2,886] rpm |
| `Torque` | Continuous | Nm | Cutting spindle drive torque | [3.8, 76.6] Nm |
| `Tool_wear` | Continuous | minutes [min] | Cumulative cutting time of active tool | [0, 253] min |
| `Type_L` | Binary (0/1) | Categorical | Low-quality machine variant (50% prevalence) | {0, 1} |
| `Type_M` | Binary (0/1) | Categorical | Medium-quality machine variant (30% prevalence) | {0, 1} |
| `Temperature_difference` | Continuous | Kelvin [K] | Engineered thermal gradient: Delta T = Process Temp - Air Temp | [7.6, 12.1] K |
| `Tool_wear_level` | Ordinal (0, 1, 2) | Quantile-binned | Tool lifecycle stage: 0=Low (0-71 min), 1=Med (72-144 min), 2=High (145-253 min) | {0, 1, 2} |

---

## Model Evaluation & Benchmark Results

### Master 6-Model Benchmark Comparison (Held-Out Test Set: N = 2,000, 68 Failures)

All models evaluated at the operational decision threshold theta = 0.50:

| Model Architecture | Test F1-Score | Test Recall | Test Precision | Test ROC-AUC | Test AP (PR-AUC) | Brier Score | Log Loss | False Negatives (Missed) | False Positives (False Alarms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tuned XGBoost (Selected)** | **80.58%** | **82.35%** | **78.87%** | **97.58%** | **86.01%** | **0.0117** | **0.0531** | **12** | **15** |
| Random Forest | 72.15% | 83.82% | 63.33% | 97.90% | 83.93% | 0.0187 | 0.0655 | 11 | 33 |
| Baseline XGBoost | 66.29% | 85.29% | 54.21% | 97.63% | 84.50% | 0.0205 | 0.0883 | 10 | 49 |
| SVM (RBF Kernel) | 55.56% | 44.12% | 75.00% | 96.78% | 65.93% | 0.0184 | 0.0961 | 38 | 10 |
| Decision Tree | 51.16% | 80.88% | 37.41% | 84.32% | 71.51% | 0.0399 | 0.8288 | 13 | 92 |
| Logistic Regression | 26.28% | 86.76% | 15.49% | 91.39% | 40.48% | 0.1199 | 0.3774 | 9 | 322 |

### Training-Only Stratified 5-Fold Cross-Validation Leaderboard

Candidate architectures were selected based strictly on 5-fold cross-validation performance across the 8,000-sample training partition before evaluating the held-out test set:

| Model Architecture | CV F1 Mean | CV F1 Std | CV Recall Mean | CV Precision Mean | CV ROC-AUC Mean | CV AP Mean | Selection Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Tuned XGBoost** | **0.7783** | **+/-0.0276** | **0.7714** | **0.7878** | **0.9685** | **0.8087** | **WINNING ARCHITECTURE** |
| Baseline XGBoost | 0.6931 | +/-0.0507 | 0.8007 | 0.6127 | 0.9735 | 0.8075 | Baseline |
| Random Forest | 0.6876 | +/-0.0424 | 0.7564 | 0.6325 | 0.9732 | 0.7759 | Baseline |
| Decision Tree | 0.5027 | +/-0.0204 | 0.8083 | 0.3661 | 0.8635 | 0.6415 | Baseline |
| SVM (RBF Kernel) | 0.4985 | +/-0.0839 | 0.3730 | 0.7654 | 0.9573 | 0.6498 | Baseline |
| Logistic Regression | 0.2433 | +/-0.0121 | 0.8193 | 0.1429 | 0.9024 | 0.4558 | Baseline |

### Final Tuned XGBoost Hyperparameters

Optimized via `RandomizedSearchCV` (30 sampled candidates, Stratified 5-Fold CV on training partition, scoring on F1):

- `n_estimators`: **269** (ensemble capacity)
- `max_depth`: **6** (tree depth limit)
- `learning_rate` (eta): **0.142009** (gradient shrinkage)
- `min_child_weight`: **2** (leaf split noise suppression)
- `subsample`: **0.942436** (stochastic row subsampling fraction)
- `colsample_bytree`: **0.879660** (feature column subsampling fraction)
- `scale_pos_weight`: **28.5203** (analytical inverse class weight: 7,729 / 271)

### Unbiased Test Performance (Held-Out Test Set: N = 2,000, Threshold theta = 0.50)

- **Accuracy:** 98.65% (1,973 / 2,000 correct predictions)
- **Precision:** 78.87% (56 true alerts out of 71 total alerts)
- **Recall:** 82.35% (56 detected failures out of 68 actual breakdowns)
- **F1-Score:** 80.58% (harmonic mean balancing precision and recall)
- **ROC-AUC:** 97.58% (discrimination across all potential operating thresholds)
- **Average Precision (PR-AUC):** 86.01% (area under the Precision-Recall curve vs. 3.39% baseline)
- **Brier Score (MSE):** 0.0117 (strictly proper scoring rule)
- **Root Mean Squared Error (RMSE):** 0.1081
- **Cross-Entropy Log Loss:** 0.0531
- **Specificity:** 99.22% (1,917 true negatives out of 1,932 normal cycles)
- **Balanced Accuracy:** 90.79%
- **Matthews Correlation Coefficient (MCC):** 0.7990
- **Confusion Matrix:**
  $$\begin{bmatrix} \text{TN} = 1917 & \text{FP} = 15 \\ \text{FN} = 12 & \text{TP} = 56 \end{bmatrix}$$
- **False Alarm Reduction:** Tuning reduced false positives from 49 down to 15 (**69.4% reduction in false alarms** compared to Baseline XGBoost), lifting Precision from 54.21% to 78.87% and F1 from 66.29% to 80.58%.

---

## Stage-Wise Boosting Diagnostics & Overfitting Analysis

### 1. Log Loss Learning Curve (Probabilistic Convergence)
- Evaluated on an internal 80/20 stratified split strictly within the training partition ($N_{\text{sub\_train}} = 6,400$, $N_{\text{val}} = 1,600$), guaranteeing zero test-set leakage.
- **Sub-Training Loss:** Drops monotonically from $0.5856 \rightarrow 0.0029$.
- **Validation Loss:** Decreases rapidly from $0.5886$ to reach its global minimum of **$0.0586$ at Round 114**, before drifting slightly upward to **$0.0652$ at Round 269**.
- **Overfitting Interpretation:** The upward drift beyond Round 114 reflects mild late-round overfitting to residual training noise. However, validation loss remains tightly bounded between $0.0586$ and $0.0652$ without diverging, confirming that row subsampling (`subsample=0.942`) and feature subsampling (`colsample_bytree=0.880`) successfully stabilize variance.

### 2. Classification Metric Learning Curves (Accuracy, Precision, Recall, F1)
- Evaluated across all 269 sequential boosting rounds at the operational decision threshold $\theta = 0.50$:
  - **Round 1 (Tree 1):** Sub-Train F1 = 0.4828 | Val F1 = 0.3962 (Val Precision = 26.58%, Val Recall = 77.78%).
  - **Round 168 (Peak Validation F1):** Val F1 reaches **0.7379** (Val Accuracy = 98.31%, Val Precision = 77.55%, Val Recall = 70.37%).
  - **Round 269 (Final Ensemble):** Sub-training metrics reach near-perfection (Sub-Train Accuracy = 99.98%, F1 = 0.9977), while validation F1 softens slightly to 0.7071 (Val Accuracy = 98.19%, Val Precision = 77.78%, Val Recall = 64.81%).
- **Methodological Governance:** The 269-estimator model configuration was finalized via full Stratified 5-Fold Cross-Validation across the entire training partition (achieving CV F1 = $0.7783 \pm 0.0276$). Early stopping on a single internal validation split was not applied post-hoc, maintaining methodological consistency.

---

## Methodological Explanations

1. **Boosting Rounds vs. Neural Network Epochs:**
   Unlike deep neural networks where continuous weight matrices are iteratively adjusted via backpropagation across epochs of the full dataset, XGBoost constructs an additive forward stage-wise ensemble:
   $$f_M(x) = \sum_{m=1}^M f_m(x)$$
   where each sequential *boosting round* $m \in \{1, \dots, 269\}$ adds an independent shallow regression tree $f_m(x)$ parameterized to fit the negative gradient (pseudo-residuals) of the cross-entropy loss function.

2. **Why R-Squared is Inappropriate for Binary Classification:**
   The coefficient of determination $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ assumes continuous, unbounded responses under Gaussian noise distributions. In binary classification with rare failure events ($3.39\%$), the true outcome $y_i \in \{0, 1\}$ is discrete, predicted probabilities $\hat{p}_i \in [0, 1]$ are bounded, and total variance $SS_{\text{tot}} = N \bar{y}(1 - \bar{y}) \approx 0.03275 \cdot N$ is minuscule. This makes $R^2$ hypersensitive and mathematically degenerate (often negative) even for superior classifiers. Instead, statistical decision theory mandates evaluating strictly proper scoring rules: **Brier Score** ($0.0117$) and **Cross-Entropy Log Loss** ($0.0531$).

3. **Separation of Held-Out Test Set from Model Selection:**
   The 2,000-sample test set (20% stratified holdout) was sealed at the start of Phase 1 and remained untouched throughout model development, cross-validation, hyperparameter tuning, and threshold selection. This guarantees an unbiased generalization estimate free from data snooping or optimistic selection bias.

---

## Explainable AI (SHAP)

To provide transparency for maintenance technicians, Phase 2 applied **SHAP (Shapley Additive Explanations)** using `shap.TreeExplainer` on the finalized Tuned XGBoost model.

### Global Normalized Feature Attribution (Held-Out Test Set)

| Rank | Sensor Feature | Mean \|SHAP Value\| | Normalized Attribution (%) | Cumulative Attribution (%) | Physical Engineering Interpretation |
|:---:|---|:---:|:---:|:---:|---|
| 1 | `Torque` | 3.3734 | **31.48%** | 31.48% | Primary mechanical overload driver; excessive cutting resistance |
| 2 | `Tool_wear` | 2.6143 | **24.40%** | 55.88% | Cutting edge dulling and progressive abrasive tool degradation |
| 3 | `Rotational_speed` | 1.6011 | **14.94%** | **70.82%** | Bifurcated risk: bearing over-speed (>2400 rpm) & stall (<1400 rpm) |
| 4 | `Temperature_difference` | 0.9764 | **9.11%** | 79.93% | Heat dissipation failure gradient: Delta T = Process Temp - Air Temp |
| 5 | `Process_temperature` | 0.7121 | **6.64%** | 86.57% | Localized thermal accumulation in cutting zone |
| 6 | `Air_temperature` | 0.7001 | **6.53%** | 93.10% | Ambient environmental temperature |
| 7 | `Tool_wear_level` | 0.3747 | **3.50%** | 96.60% | Quantile-binned tool lifecycle indicator |
| 8 | `Type_L` | 0.2302 | **2.15%** | 98.75% | Low-quality equipment variant indicator |
| 9 | `Type_M` | 0.1344 | **1.25%** | 100.00% | Medium-quality equipment variant indicator |

> **Key Finding:** The top three physical sensor drivers &mdash; **Drive Torque (31.48%)**, **Cumulative Tool Wear (24.40%)**, and **Rotational Speed (14.94%)** &mdash; account for exactly **70.82%** of total model attribution.

### Distinguishing SHAP Attribution from XGBoost Tree Gain
- **SHAP Values (Game-Theoretic Marginal Contribution):** Evaluated across individual test cycles in log-odds space using `shap.TreeExplainer`. Continuous `Torque` (31.48%) and `Tool_wear` (24.40%) dominate predictive log-odds.
- **XGBoost Gain (Split-Impurity Reduction):** Measures fractional reduction in splitting criteria across all internal nodes during training. In tree gain, `Tool_wear_level` (27.32%), `Rotational_speed` (21.74%), and `Torque` (17.30%) lead split decisions.

---

## Flask Web Application & REST API

The deployed prototype serves human technicians and automated SCADA/PLC middleware via dual operational interfaces:

### Operational Risk Categorization

| Risk Level | Failure Probability Range | Dashboard Badge | Operational Maintenance Advisory |
|:---:|:---:|:---:|---|
| **Low Risk** | p < 20% | Green Badge | Normal operating status. Maintain routine preventative inspection schedule. |
| **Medium Risk** | 20% <= p < 50% | Amber Badge | Elevated mechanical/thermal stress. Flag machine for inspection at next shift turnover. |
| **High Risk** | p >= 50% | Red Banner | **MAINTENANCE REQUIRED**: Urgent failure risk. Immediately pause machining and inspect cutting tool. |

### Two-Level Input Validation Architecture
1. **Level 1 &mdash; Hard Physical Sanity Limits (HTTP 400 Rejection):**
   Rejects physically impossible or destructive sensor telemetry prior to pipeline inference:
   - Air Temperature: $200 \le T_{\text{air}} \le 400\text{ K}$
   - Process Temperature: $200 \le T_{\text{process}} \le 450\text{ K}$
   - Rotational Speed: $100 \le \omega \le 6,000\text{ rpm}$
   - Spindle Torque: $0 \le \tau \le 150\text{ Nm}$
   - Tool Wear: $0 \le W \le 500\text{ min}$
   - Machine Variant: Must be `L`, `M`, or `H`
2. **Level 2 &mdash; Model Training Distribution Warning (Advisory Amber Alert):**
   If inputs fall within physical sanity bounds but exceed observed training ranges (e.g. $\tau \notin [3.8, 76.6]\text{ Nm}$ or $W \notin [0, 253]\text{ min}$), the application completes inference but alerts operators: *"Input values outside model training distribution; predictions may have reduced reliability."*

### Programmatic REST API Endpoints

#### 1. Retrieve Model Specifications & Benchmark Metrics
```http
GET /api/metadata
```
**Example Response:**
```json
{
  "model_name": "Tuned XGBoost Classifier",
  "operating_threshold": 0.50,
  "cv_metrics": {
    "f1_mean": 0.7783,
    "f1_std": 0.0276,
    "recall_mean": 0.7714,
    "precision_mean": 0.7878,
    "roc_auc_mean": 0.9685
  },
  "test_metrics": {
    "accuracy": 0.9865,
    "precision": 0.7887,
    "recall": 0.8235,
    "f1": 0.8058,
    "roc_auc": 0.9758,
    "average_precision": 0.8601,
    "brier_score": 0.0117,
    "log_loss": 0.0531
  }
}
```

#### 2. Execute Machine Failure Risk Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "air_temperature": 309.0,
  "process_temperature": 310.0,
  "rotational_speed": 1400.0,
  "torque": 68.0,
  "tool_wear": 230.0,
  "machine_type": "L"
}
```
> *Note: Both `"machine_type"` and `"type"` (case-insensitive: `"L"`, `"M"`, `"H"`) are supported as machine variant keys.*

**Example Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "failure_prediction": 1,
  "failure_probability": 0.9991,
  "failure_percentage": 99.91,
  "risk_level": "HIGH",
  "operational_threshold": 0.50,
  "threshold_applied": 0.50,
  "training_warning": true,
  "warnings": [
    "Air Temperature (309.0 K) outside observed training distribution [295.3, 304.5 K]"
  ],
  "warning_count": 1
}
```

---

## Technology Stack

- **Runtime Environment:** Python 3.11.x
- **Machine Learning & Preprocessing:** scikit-learn 1.5+, XGBoost 2.1+
- **Model Explainability (XAI):** SHAP 0.46+
- **Data Manipulation & Mathematics:** pandas 2.2+, NumPy 1.26+, SciPy 1.11+
- **Web Application & WSGI Server:** Flask 3.0+, Gunicorn 23.0+
- **Data Visualization:** Matplotlib 3.9+, Seaborn 0.13+
- **Automated Verification:** pytest 8.0+ / unittest
- **Cloud Hosting Platform:** Render (Managed Web Service)

---

## Setup, Execution & Testing

### 1. Clone the Repository
```bash
git clone https://github.com/S-Dinesh-Reddy/predictive-maintenance-failure-risk-assessment.git
cd predictive-maintenance-failure-risk-assessment
```

### 2. Create and Activate Virtual Environment
**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Production and inference dependencies
pip install -r requirements.txt

# Or include testing and development tools
pip install -r requirements-dev.txt
```

### 4. Run the Automated Test Suite
```bash
# Run via pytest test runner
python -m pytest tests/test_pipeline_and_app.py -v

# Or run standalone without pytest
python tests/test_pipeline_and_app.py
```
*Executes all 6 automated test suites, asserting artifact existence, dataset integrity, scaler training mean provenance (`true_mean`), pipeline inference parity, Flask HTTP/REST response contracts, Level 2 training distribution warnings, parameter alias fallback, and metadata schema alignment.*

### 5. Launch the Local Flask Application
```bash
python app.py
```
*Access the local web dashboard at `http://127.0.0.1:5000`.*

### 6. Execute Jupyter Notebooks from Clean Kernel
```bash
# Phase 1 Data Preprocessing & Exploratory Analysis
python -m nbconvert --to notebook --execute --output Phase_1_Run.ipynb notebooks/Phase_1_Data_Preprocessing.ipynb

# Phase 2 Model Training, Tuning, Learning Curves & SHAP Explainability
python -m nbconvert --to notebook --execute --output Phase_2_Run.ipynb notebooks/Phase_2_Model_Training_Evaluation.ipynb
```

---

## Deployment Configuration

The application is deployed on Render's managed cloud platform using Gunicorn as the production WSGI HTTP server:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- **Environment:** Python 3.11
- **Deployment Documentation:** Refer to [`docs/Deployment_Documentation.pdf`](docs/Deployment_Documentation.pdf) for the complete cloud runbook, architecture topology, Gunicorn tuning, and troubleshooting procedures.

---

## Project Documentation Inventory

- **Main Technical Project Report (32 Pages):** [`docs/Predictive_Maintenance_Project_Report.pdf`](docs/Predictive_Maintenance_Project_Report.pdf) &mdash; Authoritative, figure-rich documentation covering all 52 figures, theoretical derivations, learning curves, and benchmark tables.
- **Production Deployment Reference (6 Pages):** [`docs/Deployment_Documentation.pdf`](docs/Deployment_Documentation.pdf) &mdash; Detailed cloud configuration, Gunicorn setup, API schemas, and troubleshooting runbook.
- **Notebooks Workflow Reference:** [`notebooks/README.md`](notebooks/README.md) &mdash; Cell-by-cell execution audit and stage documentation.

---

## Limitations & Responsible Use

1. **Synthetic Data Origin:** The AI4I 2020 dataset was generated via physical mathematical modeling of CNC milling machines. While capturing authentic thermodynamic and mechanical correlations, it inherently lacks true factory sensor drift, electrical noise, and workpiece material hardness anomalies.
2. **Absence of High-Frequency Vibration Telemetry:** Real-world cutting tool condition monitoring relies heavily on piezoelectric accelerometer telemetry (10-20 kHz sampling) to detect cutting-edge micro-chipping hours before torque or thermal deviations occur.
3. **Static Snapshots vs. Continuous Time-Series RUL:** The dataset provides independent 1-minute operational snapshots without chronological timestamps, precluding sequential time-series forecasting (LSTM, GRU, Transformers) of continuous Remaining Useful Life (RUL).
4. **Advisory Prototype Scope:** The application functions strictly as an operator decision-support prototype. It does not interface directly with industrial programmable logic controllers (PLCs), physical safety relays, or emergency-stop (E-stop) hardware circuits.

---

## Future Improvements

1. **High-Frequency Vibration Telemetry Ingestion:** Integrate high-frequency accelerometer streams with FFT and wavelet spectral decomposition to detect micro-flaking prior to torque escalation.
2. **Sequential Time-Series Forecasting:** Transition from single-snapshot classification to sequential multivariate modeling (e.g. Temporal Convolutional Networks or Transformers) to estimate continuous Remaining Useful Life (RUL).
3. **Edge Deployment & Microcontroller Inference:** Quantize the gradient boosting ensemble into ONNX or C-based decision trees for low-latency inference directly on industrial edge gateways.
4. **Automated Drift Detection & Retraining:** Implement automated monitoring (e.g., Evidently AI) to detect concept drift and trigger retraining when machine telemetry shifts across operating seasons.

---

## References & Attribution

1. **AI4I 2020 Dataset:** Matzka, S. (2020). "Explainable Artificial Intelligence for Predictive Maintenance Applications." *Third International Conference on Artificial Intelligence for Industries (AI4I)*, pp. 69-74. IEEE.
2. **UCI Machine Learning Repository:** AI4I 2020 Predictive Maintenance Dataset. [https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
3. **SHAP (Shapley Additive Explanations):** Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, pp. 4765-4774.
4. **XGBoost:** Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, pp. 785-794.
