from pathlib import Path

from flask import Flask, request, jsonify
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fraud_model.pkl"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    text = payload.get("text") or payload.get("message")
    if not text:
        return jsonify({"error": "Missing 'text' or 'message' field in JSON payload."}), 400

    prediction = model.predict([text])[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba([text])[0][1])

    label = "fraud" if prediction == 1 else "legit"
    result = {"prediction": label}
    if probability is not None:
        result["fraud_probability"] = round(probability, 4)
    return jsonify(result)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model": str(MODEL_PATH)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
