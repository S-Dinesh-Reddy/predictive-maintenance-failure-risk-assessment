# Machine Learning-Based Predictive Maintenance and Failure Risk Assessment

An end-to-end machine learning project for predicting industrial equipment failure risk using operating-condition data from the **AI4I 2020 Predictive Maintenance Dataset**.

This project implements a leakage-safe machine learning workflow, compares multiple classification models, uses SHAP for explainability, and deploys a Flask-based predictive maintenance dashboard.

## 🚀 Live Demo

- **Live Dashboard:** [https://predictive-maintenance-failure-risk.onrender.com](https://predictive-maintenance-failure-risk.onrender.com)
- **GitHub Repository:** [predictive-maintenance-failure-risk-assessment](https://github.com/S-Dinesh-Reddy/predictive-maintenance-failure-risk-assessment)

> The application is hosted on Render's free tier. It may take a short time to start after a period of inactivity.

## 📌 Project Architecture & Workflow

The project is organized into three major phases:

### Phase 1: Data Preprocessing

- Removes identifier columns such as `UDI` and `Product ID`.
- Eliminates target-derived failure-mode columns to prevent data leakage.
- Encodes machine type using one-hot encoding.
- Engineers domain features:
  - Temperature difference
  - Tool wear level
- Performs a stratified 80/20 train-test split.
- Fits the `StandardScaler` using training data only.

### Phase 2: Model Training & Evaluation

Five baseline algorithms are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (RBF)
- Baseline XGBoost

The workflow includes:

- Stratified 5-fold cross-validation
- Hyperparameter tuning using `RandomizedSearchCV`
- Training-only model selection
- Out-of-fold threshold analysis
- Final evaluation on an untouched test set
- Probability calibration diagnostics
- SHAP-based model explainability

### Phase 3: Production Export & Deployment

- Exports the final inference pipeline and model artifacts.
- Deploys a Flask web dashboard.
- Provides prediction API endpoints.
- Includes input validation and training-distribution warnings.
- Displays machine failure probability and risk categories.
- Integrates SHAP-based diagnostic insights.

## 📊 Dataset & Feature Schema

- **Dataset:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** UCI Machine Learning Repository
- **Total Records:** 10,000
- **Target Variable:** `Machine failure`
- **Task:** Binary classification
- **Failure Rate:** Approximately 3.4%

### Features Used

The model uses nine features:

| Feature | Description |
|---|---|
| `Air_temperature` | Air temperature around the equipment |
| `Process_temperature` | Operating process temperature |
| `Rotational_speed` | Spindle rotational speed |
| `Torque` | Spindle torque |
| `Tool_wear` | Tool wear duration |
| `Type_L` | Low-quality machine variant indicator |
| `Type_M` | Medium-quality machine variant indicator |
| `Temperature_difference` | Process temperature minus air temperature |
| `Tool_wear_level` | Categorized tool lifecycle stage |

## 🏆 Model Evaluation

Tuned XGBoost was selected based on training-only cross-validation and evaluated on the held-out test set using the documented operational threshold of **0.50**.

### Final Test Results

| Metric | Tuned XGBoost |
|---|---:|
| F1-Score | 80.58% |
| Recall | 82.35% |
| Precision | 78.87% |
| ROC-AUC | 97.58% |
| Average Precision (AP) | 86.01% |
| Brier Score | 0.0117 |
| Log Loss | 0.0531 |
| False Negatives | 12 |
| False Positives | 15 |

The test partition contains 2,000 samples, including 68 failure cases.

### Model Selection Approach

Model selection was performed using training-only 5-fold cross-validation before evaluating the held-out test set. The selected model was then assessed on the test partition to provide an unbiased estimate of performance.

The operational threshold prioritizes failure detection while balancing precision and recall.

## 🔍 Explainable AI (SHAP)

SHAP TreeExplainer is used to interpret the selected Tuned XGBoost model.

The explainability analysis includes:

- Global SHAP beeswarm summary
- Feature importance ranking
- Local waterfall analysis
- Feature dependence and interaction analysis

The leading feature importance order is:

`Torque` → `Tool_wear` → `Rotational_speed` → `Air_temperature` → `Process_temperature`

SHAP values represent model feature attributions and should not be interpreted as proof of physical causation.

## 💻 Flask Web Application

The dashboard provides a predictive maintenance interface for entering equipment operating conditions and assessing failure risk.

### Risk Categorization

| Risk Level | Failure Probability | Interpretation |
|---|---:|---|
| Low Risk | < 20% | Normal Status |
| Medium Risk | 20% to < 50% | Elevated Caution |
| High Risk | ≥ 50% | Maintenance Required |

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/metadata` | Returns model specifications, operational threshold, and test metrics |
| `POST /api/predict` | Accepts JSON input and returns a validated prediction |

The application includes input sanity checks and advisory warnings when inputs fall outside the observed training ranges.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Machine Learning:** scikit-learn, XGBoost
- **Data Processing:** pandas, NumPy
- **Explainable AI:** SHAP
- **Web Framework:** Flask
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Render, Gunicorn
- **Development Environment:** Jupyter Notebook

## ⚙️ Setup and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/S-Dinesh-Reddy/predictive-maintenance-failure-risk-assessment.git
cd predictive-maintenance-failure-risk-assessment
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

```bash
python app.py
```

Open the dashboard in your browser:

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

### 5. Run the Test Suite

```bash
python tests/test_pipeline_and_app.py
```

The test suite validates required artifacts, dataset integrity, scaler parameters, pipeline inference, and Flask UI/API endpoints.

### 6. Execute the Jupyter Notebooks

Run the notebooks in order:

1. `notebooks/Phase_1_Data_Preprocessing.ipynb`
2. `notebooks/Phase_2_Model_Training_Evaluation.ipynb`

The exported model artifacts in `models/` are required to run the dashboard. Re-running the notebooks is not necessary when the existing artifacts are available.

## 📚 Documentation

- [Predictive Maintenance Project Report](docs/Predictive_Maintenance_Project_Report.pdf)
- [Deployment Documentation](docs/Deployment_Documentation.pdf)

Refer to the project report for the detailed methodology and evaluation, and the deployment documentation for hosting configuration and deployment steps.

## ⚠️ Limitations & Responsible Use

1. **Synthetic Dataset:** AI4I 2020 is a synthetic benchmark. Results on this dataset do not guarantee equivalent performance on real industrial machinery.
2. **Input Extrapolation:** Predictions outside the training distribution may be unreliable. The dashboard provides advisory warnings for such inputs.
3. **Attribution vs. Causation:** SHAP explains model behavior but does not establish mechanical causation.
4. **Decision Support:** The system is intended to support maintenance decisions, not replace qualified inspection or serve as a standalone autonomous shutdown mechanism.
5. **Industrial Validation:** Site-specific testing, calibration, and validation are required before deployment in real-world operations.

---

**Developed as a machine learning project focused on predictive maintenance, failure risk assessment, model explainability, and deployment.**