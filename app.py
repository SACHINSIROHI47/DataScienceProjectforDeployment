from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Define the Flask app
app = Flask(__name__)

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Define the route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get data from request
    data = request.json
    features = data.get("features")

    # Check if features are provided
    if not features or len(features) != 4:
        return jsonify({"error": "Please provide an array of 4 features"}), 400

    # Make prediction
    prediction = model.predict([features])
    species = ["setosa", "versicolor", "virginica"]

    # Return the prediction result
    return jsonify({"prediction": species[prediction[0]]})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
