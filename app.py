from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "tuned_xgboost_model.pkl"
)

model = joblib.load(model_path)

xgb_feature_names = [
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
TRAINING_RANGES = {
    "air_temperature": {"min": 295.3, "max": 304.5, "label": "Air Temperature", "unit": "K"},
    "process_temperature": {"min": 305.7, "max": 313.8, "label": "Process Temperature", "unit": "K"},
    "rotational_speed": {"min": 1168.0, "max": 2886.0, "label": "Rotational Speed", "unit": "rpm"},
    "torque": {"min": 3.8, "max": 76.6, "label": "Torque", "unit": "Nm"},
    "tool_wear": {"min": 0.0, "max": 253.0, "label": "Tool Wear", "unit": "min"}
}

VALID_MACHINE_TYPES = {"H", "M", "L"}


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        probability=None,
        risk_level=None,
        training_warning=False,
        out_of_dist_features=[]
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
                    error=f"Invalid or malformed numeric input for {config['label']}: '{raw_val}'."
                )

            # Reject non-finite values (NaN, Inf)
            import math
            if math.isnan(val) or math.isinf(val):
                return render_template(
                    "index.html",
                    prediction=None,
                    probability=None,
                    risk_level=None,
                    training_warning=False,
                    out_of_dist_features=[],
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
                    error=f"{config['label']} ({val} {config['unit']}) is outside the supported input range ({config['min']:.0f} – {config['max']:.0f} {config['unit']})."
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
            v = validated_values[feat]
            if v < t_config["min"] or v > t_config["max"]:
                out_of_dist_warnings.append(f"{t_config['label']}: {v} {t_config['unit']}")

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

        # Construct input DataFrame with exact feature names and order used during XGBoost training
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
            columns=xgb_feature_names
        )

        # XGBoost inference on unscaled features
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1] * 100)

        # Risk level categorization
        if probability < 20:
            risk_level = "LOW"
        elif probability < 50:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability, 2),
            risk_level=risk_level,
            training_warning=training_warning,
            out_of_dist_features=out_of_dist_warnings
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
            error="Unable to process the prediction. Please check your inputs."
        )


if __name__ == "__main__":
    app.run(
        debug=True
    )