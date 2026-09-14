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


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        probability=None,
        risk_level=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        air_temperature = float(request.form["air_temperature"])
        process_temperature = float(request.form["process_temperature"])
        rotational_speed = float(request.form["rotational_speed"])
        torque = float(request.form["torque"])
        tool_wear = float(request.form["tool_wear"])
        machine_type = str(request.form.get("machine_type", "H")).strip().upper()

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
            risk_level=risk_level
        )

    except Exception as e:
        print("Prediction error:", e)
        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            risk_level=None,
            error="Unable to process the prediction. Please check your inputs."
        )


if __name__ == "__main__":
    app.run(
        debug=True
    )