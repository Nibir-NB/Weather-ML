from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import requests
import os

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Load the trained models
solar_model = joblib.load("solar.pkl")  # Solar model
wind_model = joblib.load("wind.pkl")    # Wind model

# Load API key from environment variable
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    raise ValueError("Missing OpenWeather API key. Set the OPENWEATHER_API_KEY environment variable.")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_power_from_location():
    data = request.json
    place = data.get('place')

    # Call OpenWeatherMap API
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={place}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(weather_url)
    if response.status_code != 200:
        return jsonify({"error": "Place not found or weather API failed"}), 400

    weather_data = response.json()

    # Extract required weather details
    cloud_cover = weather_data['clouds']['all']
    wind_speed = weather_data['wind']['speed']
    temperature = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    air_density = weather_data['main'].get('pressure', 1013) / (287.05 * (temperature + 273.15))
    solar_irradiance = max(0, 1000 - cloud_cover * 10)

    # Predict solar power
    solar_features = np.array([[solar_irradiance, temperature, wind_speed, humidity]])
    predicted_power_per_m2 = solar_model.predict(solar_features)[0]

    # Predict wind power
    wind_features = np.array([[wind_speed, air_density, temperature, humidity]])
    predicted_wind_pct = wind_model.predict(wind_features)[0]

    # Return predictions
    return jsonify({
        'place': place,
        'solar_irradiance': solar_irradiance,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,
        'predicted_power_per_m2': predicted_power_per_m2,
        'air_density': round(air_density, 4),
        'predicted_wind_power_pct': predicted_wind_pct
    })

# Run the app (for local testing or platform support)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
