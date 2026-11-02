import os

import joblib
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from kan import KAN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

artifacts = joblib.load(os.path.join(MODEL_DIR, "preprocessing.pkl"))

tfidf = artifacts["tfidf"]
svd = artifacts["svd"]
crime_encoder = artifacts["crime_encoder"]
legal_encoder = artifacts["legal_encoder"]
input_features = artifacts["input_features"]

crime_model = KAN(
    width=[input_features, 16, artifacts["crime_classes"]],
    grid=5,
    k=3,
    seed=42,
    device=str(DEVICE),
)
crime_model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "crime_kan.pth"),
        map_location=DEVICE,
        weights_only=True,
    )
)
crime_model.eval()

legal_model = KAN(
    width=[input_features, 16, artifacts["legal_classes"]],
    grid=5,
    k=3,
    seed=42,
    device=str(DEVICE),
)
legal_model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "legal_kan.pth"),
        map_location=DEVICE,
        weights_only=True,
    )
)
legal_model.eval()


def predict_cybercrime(description):
    text_features = tfidf.transform([description])
    reduced_features = svd.transform(text_features)

    X = torch.tensor(
        np.asarray(reduced_features),
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        crime_output = crime_model(X)
        crime_index = crime_output.argmax(dim=1).item()
        crime_prediction = crime_encoder.inverse_transform([crime_index])[0]

        legal_output = legal_model(X)
        probabilities = torch.sigmoid(legal_output).cpu().numpy()[0]

        predicted_sections = [
            label
            for label, probability in zip(legal_encoder.classes_, probabilities)
            if probability >= 0.5
        ]

    return crime_prediction, predicted_sections


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    description = request.form.get("description", "").strip()

    if not description:
        return jsonify({"error": "Please enter a cybercrime description."}), 400

    crime, sections = predict_cybercrime(description)

    return jsonify(
        {
            "crime_type": crime,
            "legal_sections": sections,
            "message": "Prediction generated successfully.",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
