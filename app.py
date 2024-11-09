from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models, scaler, and label encoder
# earthquake_clf = joblib.load('random_forest_classifier.pkl')
earthquake_clf = joblib.load('random_forest_classifier.pkl')
earthquake_reg = joblib.load('random_forest_regressor.pkl')
# earthquake_reg = joblib.load('C:\Users\user\Desktop\AI-DRIVEN-DISASTER-PREDICTION-MODEL\random_forest_regressor.pkl')
earthquake_scaler = joblib.load('scaler.pkl')
# earthquake_scaler = joblib.load('C:\Users\user\Desktop\AI-DRIVEN-DISASTER-PREDICTION-MODEL\scaler.pkl')
earthquake_le = joblib.load('label_encoder.pkl')


# Load flood model and scaler
flood_clf = joblib.load('flood_classifier.pkl')
flood_scaler = joblib.load('flood_scaler.pkl')

# Prediction API Routes
@app.route('/')
def index():
    return render_template('index.html')

# Earthquake Prediction Route
@app.route('/predict', methods=['POST', 'GET'])
def predict_earthquake():
    if request.method == 'POST':
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        depth = float(data['depth'])
        input_data = pd.DataFrame({'Lat': [latitude], 'Long': [longitude], 'Depth': [depth], 'Origin Time': [0]})
        input_data[['Lat', 'Long', 'Depth', 'Origin Time']] = earthquake_scaler.transform(input_data[['Lat', 'Long', 'Depth', 'Origin Time']])

        predicted_category_encoded = earthquake_clf.predict(input_data)
        predicted_category = earthquake_le.inverse_transform(predicted_category_encoded)
        predicted_magnitude = earthquake_reg.predict(input_data)

        return jsonify({
            'predicted_category': predicted_category[0],
            'predicted_magnitude': round(predicted_magnitude[0], 2)
        })
    return render_template('earthquake_index.html')

# Flood Prediction Route
@app.route('/predict_flood', methods=['POST', 'GET'])
def predict_flood():
    if request.method == 'POST':
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        rainfall = float(data['rainfall'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        river_discharge = float(data['river_discharge'])
        water_level = float(data['water_level'])
        elevation = float(data['elevation'])

        # Prepare input data for flood prediction
        input_data = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Rainfall': [rainfall],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'River Discharge': [river_discharge],
            'Water Level': [water_level],
            'Elevation': [elevation]
        })

        # Scale input data
        input_data_scaled = flood_scaler.transform(input_data)

        # Predict flood risk
        flood_prediction = flood_clf.predict(input_data_scaled)
        flood_result = "Flood likely" if flood_prediction[0] == 1 else "Flood unlikely"

        return jsonify({'flood_result': flood_result})
    return render_template('flood_index.html')

# Weather Route
@app.route('/weather', methods=['GET'])
def weather():
    return render_template('weather.html')

if __name__ == '__main__':
    app.run(debug=True)