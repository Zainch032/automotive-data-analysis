"""Small web view of the car-price notebook: findings, problems, and a dollar estimate."""

from flask import Flask, jsonify, render_template, request

from car_model import get_state, predict

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", page=get_state()["page"], options=get_state()["options"])


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(predict(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    print("Training the price model on Car.csv...")
    get_state()
    print("Open http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)
