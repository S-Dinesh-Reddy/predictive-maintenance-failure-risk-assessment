from flask import Flask, render_template, request

app = Flask(__name__)


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

    # Receive machine parameters from the frontend
    air_temperature = float(request.form["air_temperature"])
    process_temperature = float(request.form["process_temperature"])
    rotational_speed = float(request.form["rotational_speed"])
    torque = float(request.form["torque"])
    tool_wear = float(request.form["tool_wear"])

    # Temporary placeholder until the ML model is trained
    prediction = None
    probability = None
    risk_level = None

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk_level=risk_level
    )


if __name__ == "__main__":
    app.run(debug=True)