"""EV Charge Time Predictor: Flask web app.

Run:  python app.py      then open http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request

import ml

app = Flask(__name__)

INPUT_RANGES = {            # name: (min, max)
    "battery": (10, 200),
    "start": (0, 95),
    "target": (5, 100),
    "charger": (1, 350),
    "temp": (-30, 60),
}


def read_int(value, name, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return number


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/summary")
def api_summary():
    try:
        depth = int(read_int(request.args.get("depth", 8), "depth", 1, 20))
        leaf = int(read_int(request.args.get("leaf", 5), "leaf", 1, 100))
        return jsonify(ml.summary(depth, leaf))
    except ValueError as err:
        return jsonify(error=str(err)), 400


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    try:
        values = {k: read_int(body.get(k), k, lo, hi) for k, (lo, hi) in INPUT_RANGES.items()}
        if values["target"] < values["start"] + 5:
            raise ValueError("target must be at least 5 points above start")
        depth = int(read_int(body.get("depth", 8), "depth", 1, 20))
        leaf = int(read_int(body.get("leaf", 5), "leaf", 1, 100))
        return jsonify(ml.predict(
            values["battery"], values["start"], values["target"],
            values["charger"], values["temp"], depth, leaf))
    except ValueError as err:
        return jsonify(error=str(err)), 400


if __name__ == "__main__":
    ml.load_data()          # fail early if the data file has a problem
    app.run(debug=True)
