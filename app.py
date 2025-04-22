import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
model_filename = "model5.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return "ML Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Ensure all expected columns exist, add default values if missing
        expected_columns = ["ASN", "Login Hour", "IP Address", "User Agent String",
                            "Browser Name and Version", "OS Name and Version",
                            "Country", "Device Type"]

        default_values = {
            "Country": "Unknown",
            "Device Type": "Other"
        }

        for col in expected_columns:
            if col not in data:
                data[col] = default_values.get(col, 0)

        # Convert input into DataFrame
        input_data = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
