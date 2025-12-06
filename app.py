from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

autoencoder = load_model("autoencoder_model.h5")
encoder = load_model("encoder_model.h5")

scaler = joblib.load("scaler.pkl")
product_columns = joblib.load("product_columns.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(type(data['user_item_matrix']), data['user_item_matrix'])

    user_vector = np.array(data["user_item_matrix"])  # dạng [[0,1,0,2,...]]
    
    # Scale
    user_scaled = scaler.transform(user_vector)

    # Predict
    encoded = encoder.predict(user_scaled).tolist()

    return jsonify({
        "encoded_data": encoded
    })

if __name__ == "__main__":
    app.run(debug=True)
