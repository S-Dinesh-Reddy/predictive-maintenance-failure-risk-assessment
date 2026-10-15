from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
import os
import math

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load Production Artifacts
pipeline_path = os.path.join(BASE_DIR, "models", "final_pipeline.pkl")
metadata_path = os.path.join(BASE_DIR, "models", "model_metadata.json")

if os.path.exists(pipeline_path):
    model_pipeline = joblib.load(pipeline_path)
else:
    model_pipeline = joblib.load(os.path.join(BASE_DIR, "models", "tuned_xgboost_model.pkl"))

metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

OPERATIONAL_THRESHOLD = float(metadata.get("operating_threshold", 0.50))
TEST_METRICS = metadata.get("test_metrics", {
    "f1": 0.8058,
    "recall": 0.8235,
    "precision": 0.7887,
    "roc_auc": 0.9758,
    "accuracy": 0.9865
})

feature_names = [
    "Air_temperature",
    "Process_temperature",
    "Rotational_speed",
    "Torque",
    "Tool_wear",
    "Type_L",
    "Type_M",
    "Temperature_difference",
    "Tool_wear_level"
]

# Level 1 — Hard Application Sanity Limits (Prevents impossible/nonsensical inputs)
SANITY_RANGES = {
    "air_temperature": {"min": 200.0, "max": 400.0, "label": "Air Temperature", "unit": "K"},
    "process_temperature": {"min": 200.0, "max": 450.0, "label": "Process Temperature", "unit": "K"},
    "rotational_speed": {"min": 100.0, "max": 6000.0, "label": "Rotational Speed", "unit": "rpm"},
    "torque": {"min": 0.0, "max": 150.0, "label": "Torque", "unit": "Nm"},
    "tool_wear": {"min": 0.0, "max": 500.0, "label": "Tool Wear", "unit": "min"}
}

# Level 2 — AI4I 2020 Observed Model Training Distribution (For soft extrapolation warnings)
TRAINING_RANGES = metadata.get("training_distributions", {
    "air_temperature": {"min": 295.3, "max": 304.5, "label": "Air Temperature", "unit": "K"},
    "process_temperature": {"min": 305.7, "max": 313.8, "label": "Process Temperature", "unit": "K"},
    "rotational_speed": {"min": 1168.0, "max": 2886.0, "label": "Rotational Speed", "unit": "rpm"},
    "torque": {"min": 3.8, "max": 76.6, "label": "Torque", "unit": "Nm"},
    "tool_wear": {"min": 0.0, "max": 253.0, "label": "Tool Wear", "unit": "min"}
})

VALID_MACHINE_TYPES = {"H", "M", "L"}


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        probability=None,
        risk_level=None,
        training_warning=False,
        out_of_dist_features=[],
        metrics=TEST_METRICS,
        threshold=OPERATIONAL_THRESHOLD,
        metadata=metadata
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Validate Machine Type
        machine_type = str(request.form.get("machine_type", "")).strip().upper()
        if machine_type not in VALID_MACHINE_TYPES:
            return render_template(
                "index.html",
                prediction=None,
                probability=None,
                risk_level=None,
                training_warning=False,
                out_of_dist_features=[],
                metrics=TEST_METRICS,
                threshold=OPERATIONAL_THRESHOLD,
                metadata=metadata,
                error="Invalid Machine Type. Must be H (High), M (Medium), or L (Low)."
            )

        # 2. Level 1 Hard Sanity Validation
        validated_values = {}
        for feat, config in SANITY_RANGES.items():
            raw_val = request.form.get(feat, "").strip()
            if not raw_val:
                return render_template(
                    "index.html",
                    prediction=None,
                    probability=None,
                    risk_level=None,
                    training_warning=False,
                    out_of_dist_features=[],
                    metrics=TEST_METRICS,
                    threshold=OPERATIONAL_THRESHOLD,
                    metadata=metadata,
                    error=f"Missing value for {config['label']}."
                )

            # Reject malformed strings
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                return render_template(
                    "index.html",
                    prediction=None,
                    probability=None,
                    risk_level=None,
                    training_warning=False,
                    out_of_dist_features=[],
                    metrics=TEST_METRICS,
                    threshold=OPERATIONAL_THRESHOLD,
                    metadata=metadata,
                    error=f"Invalid or malformed numeric input for {config['label']}: '{raw_val}'."
                )

            # Reject non-finite values (NaN, Inf)
            if math.isnan(val) or math.isinf(val):
                return render_template(
                    "index.html",
                    prediction=None,
                    probability=None,
                    risk_level=None,
                    training_warning=False,
                    out_of_dist_features=[],
                    metrics=TEST_METRICS,
                    threshold=OPERATIONAL_THRESHOLD,
                    metadata=metadata,
                    error=f"Non-finite numeric value provided for {config['label']}."
                )

            # Hard Sanity Boundary Check (rejects negative / physically impossible values)
            if val < config["min"] or val > config["max"]:
                return render_template(
                    "index.html",
                    prediction=None,
                    probability=None,
                    risk_level=None,
                    training_warning=False,
                    out_of_dist_features=[],
                    metrics=TEST_METRICS,
                    threshold=OPERATIONAL_THRESHOLD,
                    metadata=metadata,
                    error=f"{config['label']} ({val} {config['unit']}) is outside supported operating bounds ({config['min']:.0f} – {config['max']:.0f} {config['unit']})."
                )

            validated_values[feat] = val

        air_temperature = validated_values["air_temperature"]
        process_temperature = validated_values["process_temperature"]
        rotational_speed = validated_values["rotational_speed"]
        torque = validated_values["torque"]
        tool_wear = validated_values["tool_wear"]

        # 3. Level 2 Training Distribution Check (generates advisory warning, does NOT reject)
        out_of_dist_warnings = []
        for feat, t_config in TRAINING_RANGES.items():
            v = validated_values.get(feat)
            if v is not None and (v < t_config["min"] or v > t_config["max"]):
                label = SANITY_RANGES[feat]["label"]
                out_of_dist_warnings.append(f"{label}: {v} {t_config['unit']}")

        training_warning = len(out_of_dist_warnings) > 0

        # Engineered features matching Phase 1 training preprocessing
        temperature_difference = process_temperature - air_temperature

        # Tool wear level quantiles: <=71: Low (0), 72-144: Medium (1), >144: High (2)
        if tool_wear <= 71:
            tool_wear_level = 0
        elif tool_wear <= 144:
            tool_wear_level = 1
        else:
            tool_wear_level = 2

        # Machine Type One-Hot Encoding (Reference category: H)
        if machine_type == "L":
            type_l = 1
            type_m = 0
        elif machine_type == "M":
            type_l = 0
            type_m = 1
        else:
            type_l = 0
            type_m = 0

        # Construct input DataFrame with exact feature names and order
        input_data = pd.DataFrame(
            [[
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
                type_l,
                type_m,
                temperature_difference,
                tool_wear_level
            ]],
            columns=feature_names
        )

        # Inference through the production pipeline
        probability = float(model_pipeline.predict_proba(input_data)[0][1] * 100)
        prediction = 1 if (probability >= OPERATIONAL_THRESHOLD * 100.0) else 0

        # Risk level categorization based on documented tiers
        if probability < 20.0:
            risk_level = "LOW"
        elif probability < 50.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability, 2),
            risk_level=risk_level,
            training_warning=training_warning,
            out_of_dist_features=out_of_dist_warnings,
            metrics=TEST_METRICS,
            threshold=OPERATIONAL_THRESHOLD,
            metadata=metadata
        )

    except Exception as e:
        print("Prediction error:", e)
        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            risk_level=None,
            training_warning=False,
            out_of_dist_features=[],
            metrics=TEST_METRICS,
            threshold=OPERATIONAL_THRESHOLD,
            metadata=metadata,
            error="Unable to process the prediction. Please verify your sensor inputs."
        )


@app.route("/api/metadata", methods=["GET"])
def api_metadata():
    """Return model specifications, metrics, and training bounds."""
    return jsonify({
        "status": "success",
        "metadata": metadata,
        "operational_threshold": OPERATIONAL_THRESHOLD,
        "test_metrics": TEST_METRICS
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Programmatic JSON prediction endpoint for automated validation."""
    try:
        data = request.get_json(force=True)
        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "No JSON payload provided", "error": "No JSON payload provided"}), 400

        required_fields = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "machine_type"]
        for field in required_fields:
            if field not in data:
                err_msg = f"Missing required parameter: {field}"
                return jsonify({"status": "error", "message": err_msg, "error": err_msg}), 400

        machine_type = str(data["machine_type"]).strip().upper()
        if machine_type not in VALID_MACHINE_TYPES:
            err_msg = "Invalid machine_type (must be H, M, or L)"
            return jsonify({"status": "error", "message": err_msg, "error": err_msg}), 400

        # Validate numeric types and sanity limits
        for field, limits in SANITY_RANGES.items():
            try:
                val = float(data[field])
            except (ValueError, TypeError):
                err_msg = f"Invalid numeric value for {limits['label']}"
                return jsonify({"status": "error", "message": err_msg, "error": err_msg}), 400

            if math.isnan(val) or math.isinf(val) or val < limits["min"] or val > limits["max"]:
                err_msg = f"{limits['label']} ({val} {limits['unit']}) is outside supported operating bounds [{limits['min']}, {limits['max']} {limits['unit']}]"
                return jsonify({"status": "error", "message": err_msg, "error": err_msg}), 400

        air_temp = float(data["air_temperature"])
        proc_temp = float(data["process_temperature"])
        rot_speed = float(data["rotational_speed"])
        torque = float(data["torque"])
        tool_wear = float(data["tool_wear"])

        temp_diff = proc_temp - air_temp
        tool_wear_level = 0 if tool_wear <= 71 else (1 if tool_wear <= 144 else 2)
        type_l = 1 if machine_type == "L" else 0
        type_m = 1 if machine_type == "M" else 0

        input_data = pd.DataFrame(
            [[air_temp, proc_temp, rot_speed, torque, tool_wear, type_l, type_m, temp_diff, tool_wear_level]],
            columns=feature_names
        )

        prob = float(model_pipeline.predict_proba(input_data)[0][1] * 100)
        pred = 1 if prob >= OPERATIONAL_THRESHOLD * 100.0 else 0
        risk = "LOW" if prob < 20.0 else ("MEDIUM" if prob < 50.0 else "HIGH")

        return jsonify({
            "status": "success",
            "prediction": pred,
            "failure_probability": round(prob, 4),
            "risk_level": risk,
            "threshold_applied": OPERATIONAL_THRESHOLD
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)