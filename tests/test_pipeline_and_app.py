"""
Automated Test Suite for Predictive Maintenance Pipeline, Scaler Provenance, and Flask Integration
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


def test_artifacts_exist():
    """Verify that all required production artifacts are present and non-empty."""
    expected_files = [
        "final_pipeline.pkl",
        "tuned_xgboost_model.pkl",
        "scaler.pkl",
        "model_metadata.json"
    ]
    for filename in expected_files:
        path = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(path), f"Missing artifact: {filename}"
        assert os.path.getsize(path) > 0, f"Artifact is empty: {filename}"


def test_dataset_integrity():
    """Verify dataset row counts, schemas, target alignment, and no NaNs."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

    assert len(X_train) == 8000, f"Expected 8000 training rows, got {len(X_train)}"
    assert len(X_test) == 2000, f"Expected 2000 test rows, got {len(X_test)}"
    assert len(y_train) == 8000, f"Expected 8000 train labels, got {len(y_train)}"
    assert len(y_test) == 2000, f"Expected 2000 test labels, got {len(y_test)}"

    assert X_train.shape[1] == 9, f"Expected 9 features, got {X_train.shape[1]}"
    assert X_test.shape[1] == 9, f"Expected 9 features, got {X_test.shape[1]}"

    assert set(y_train.unique()).issubset({0, 1}), "y_train target values are not binary {0, 1}"
    assert set(y_test.unique()).issubset({0, 1}), "y_test target values are not binary {0, 1}"

    # Verify zero nulls
    assert X_train.isnull().sum().sum() == 0, "X_train contains missing values"
    assert X_test.isnull().sum().sum() == 0, "X_test contains missing values"


def test_scaler_provenance():
    """Verify the scaler has the true training parameters and scales correctly."""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")
    assert len(scaler.mean_) == 9
    assert len(scaler.scale_) == 9

    # Verify Air temperature mean is around 300K, not 0
    assert 299.0 < scaler.mean_[0] < 301.0, f"Scaler mean[0] is {scaler.mean_[0]}, expected ~300 K"
    assert 1.9 < scaler.scale_[0] < 2.1, f"Scaler scale[0] is {scaler.scale_[0]}, expected ~2.0 K"

    # Verify n_samples_seen_ is 8000
    assert scaler.n_samples_seen_ == 8000, f"Expected 8000 samples seen, got {scaler.n_samples_seen_}"


def test_pipeline_inference():
    """Verify production pipeline correctly scores normal and extreme machine conditions."""
    pipeline = joblib.load(os.path.join(MODELS_DIR, "final_pipeline.pkl"))
    metadata = json.load(open(os.path.join(MODELS_DIR, "model_metadata.json"), encoding="utf-8"))
    threshold = metadata.get("operating_threshold", 0.50)

    clean_cols = metadata.get("features", metadata.get("feature_names"))

    # Normal operation
    normal_df = pd.DataFrame([[300.0, 310.0, 1400.0, 30.0, 20.0, 0, 0, 10.0, 0]], columns=clean_cols)
    prob_normal = pipeline.predict_proba(normal_df)[0][1]
    pred_normal = int(prob_normal >= threshold)
    assert pred_normal == 0, f"Normal sample predicted as failure: prob={prob_normal}"
    assert prob_normal < 0.20, f"Normal sample probability too high: {prob_normal}"

    # Extreme high-stress operation (high torque + severe tool wear)
    stress_df = pd.DataFrame([[303.0, 312.0, 1250.0, 68.0, 230.0, 1, 0, 9.0, 2]], columns=clean_cols)
    prob_stress = pipeline.predict_proba(stress_df)[0][1]
    pred_stress = int(prob_stress >= threshold)
    assert pred_stress == 1, f"Stress sample failed to predict failure: prob={prob_stress}"
    assert prob_stress >= 0.50, f"Stress sample probability too low: {prob_stress}"


def test_flask_endpoints():
    """Verify Flask application web UI and API routes."""
    import sys
    sys.path.insert(0, BASE_DIR)
    from app import app

    app.config["TESTING"] = True
    client = app.test_client()

    # 1. GET Home
    res = client.get("/")
    assert res.status_code == 200
    assert b"PREDICTIVE MAINTENANCE" in res.data
    assert b"OPERATING CONDITIONS" in res.data

    # 2. POST Normal UI form
    res_pred_normal = client.post("/predict", data={
        "air_temperature": "300.0",
        "process_temperature": "310.0",
        "rotational_speed": "1400",
        "torque": "30.0",
        "tool_wear": "20",
        "machine_type": "H"
    })
    assert res_pred_normal.status_code == 200
    assert b"NORMAL" in res_pred_normal.data
    assert b"LOW" in res_pred_normal.data

    # 3. POST High Stress UI form
    res_pred_stress = client.post("/predict", data={
        "air_temperature": "303.0",
        "process_temperature": "312.0",
        "rotational_speed": "1250",
        "torque": "68.0",
        "tool_wear": "230",
        "machine_type": "L"
    })
    assert res_pred_stress.status_code == 200
    assert b"MAINTENANCE REQUIRED" in res_pred_stress.data
    assert b"HIGH" in res_pred_stress.data

    # 4. POST Validation Rejection (Out of sanity range)
    res_invalid = client.post("/predict", data={
        "air_temperature": "150.0",  # Below min 200 K
        "process_temperature": "310.0",
        "rotational_speed": "1400",
        "torque": "30.0",
        "tool_wear": "20",
        "machine_type": "H"
    })
    assert res_invalid.status_code == 200
    assert b"outside supported operating bounds" in res_invalid.data

    # 5. GET API Metadata
    res_meta = client.get("/api/metadata")
    assert res_meta.status_code == 200
    data_meta = res_meta.get_json()
    assert data_meta["status"] == "success"
    assert "test_metrics" in data_meta
    assert data_meta["operational_threshold"] == 0.50

    # 6. POST API Predict Valid (testing standardized keys and scale)
    res_api_pred = client.post("/api/predict", json={
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1400.0,
        "torque": 30.0,
        "tool_wear": 20.0,
        "machine_type": "H"
    })
    assert res_api_pred.status_code == 200
    data_api = res_api_pred.get_json()
    assert data_api["status"] == "success"
    assert data_api["prediction"] == 0
    assert data_api["failure_prediction"] == 0
    assert data_api["risk_level"] == "LOW"
    assert "failure_probability" in data_api
    assert "failure_percentage" in data_api
    assert 0.0 <= data_api["failure_probability"] <= 1.0
    assert 0.0 <= data_api["failure_percentage"] <= 100.0
    assert abs(data_api["failure_probability"] * 100.0 - data_api["failure_percentage"]) < 0.1
    assert data_api["operational_threshold"] == 0.50
    assert data_api["threshold_applied"] == 0.50
    assert data_api["training_warning"] is False
    assert data_api["warning_count"] == 0
    assert data_api["warnings"] == []

    # 7. POST API Predict with 'type' parameter fallback (backward compatibility)
    res_api_type_fallback = client.post("/api/predict", json={
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1400.0,
        "torque": 30.0,
        "tool_wear": 20.0,
        "type": "L"  # using 'type' instead of 'machine_type'
    })
    assert res_api_type_fallback.status_code == 200
    assert res_api_type_fallback.get_json()["status"] == "success"

    # 8. POST API Predict with Level 2 Out-of-Distribution Warning
    # Torque 100.0 Nm is within physical sanity (0-150 Nm) but outside training distribution (3.8-76.6 Nm)
    res_api_ood = client.post("/api/predict", json={
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1400.0,
        "torque": 100.0,
        "tool_wear": 20.0,
        "machine_type": "M"
    })
    assert res_api_ood.status_code == 200
    data_ood = res_api_ood.get_json()
    assert data_ood["status"] == "success"
    assert data_ood["training_warning"] is True
    assert data_ood["warning_count"] >= 1
    assert any("Torque" in w for w in data_ood["warnings"])

    # 9. POST API Predict Invalid Machine Type (must reject with 400)
    res_api_bad_type = client.post("/api/predict", json={
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1400.0,
        "torque": 30.0,
        "tool_wear": 20.0,
        "machine_type": "Z"
    })
    assert res_api_bad_type.status_code == 400
    assert "Invalid machine type" in res_api_bad_type.get_json()["error"]

    # 10. POST API Predict Invalid (Missing field)
    res_api_bad = client.post("/api/predict", json={
        "air_temperature": 300.0
    })
    assert res_api_bad.status_code == 400
    data_bad = res_api_bad.get_json()
    assert data_bad["status"] == "error"
    assert "Missing required parameter" in data_bad["error"]


def test_metadata_pipeline_consistency():
    """Verify metadata definitions, schema consistency, and metric bounds against the production pipeline."""
    metadata = json.load(open(os.path.join(MODELS_DIR, "model_metadata.json"), encoding="utf-8"))
    pipeline = joblib.load(os.path.join(MODELS_DIR, "final_pipeline.pkl"))

    # Assert 9 input features
    features = metadata.get("features", metadata.get("feature_names"))
    assert len(features) == 9, f"Expected 9 features in metadata, got {len(features)}"

    # Assert operational threshold
    assert metadata.get("operating_threshold") == 0.50

    # Assert test metrics exist and match precision benchmarks
    test_m = metadata.get("test_metrics", {})
    assert test_m.get("accuracy") == 0.9865
    assert test_m.get("precision") == 0.7887
    assert test_m.get("recall") == 0.8235
    assert test_m.get("f1") == 0.8058
    assert test_m.get("roc_auc") == 0.9758
    assert test_m.get("average_precision") == 0.8601
    assert test_m.get("brier_score") == 0.0117
    assert test_m.get("log_loss") == 0.0531

    # Verify pipeline can score a test vector with exact metadata feature names
    dummy_input = pd.DataFrame([[300.0, 310.0, 1500.0, 40.0, 50.0, 0, 1, 10.0, 1]], columns=features)
    prob = pipeline.predict_proba(dummy_input)
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0][1] <= 1.0


if __name__ == "__main__":
    print("Running pipeline and Flask tests...")
    test_artifacts_exist()
    print("  test_artifacts_exist: PASSED")
    test_dataset_integrity()
    print("  test_dataset_integrity: PASSED")
    test_scaler_provenance()
    print("  test_scaler_provenance: PASSED")
    test_pipeline_inference()
    print("  test_pipeline_inference: PASSED")
    test_flask_endpoints()
    print("  test_flask_endpoints: PASSED")
    test_metadata_pipeline_consistency()
    print("  test_metadata_pipeline_consistency: PASSED")
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY!")
