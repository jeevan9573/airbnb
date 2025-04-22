from flask import Flask, render_template, request, jsonify
from geopy.distance import geodesic
import joblib

app = Flask(__name__)

# Load your one‑feature model
model = joblib.load("models/airbnb_price_predictio.pkl")

# New York city center for distance calculation
DEFAULT_CITY_CENTER = (40.7128, -74.0060)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Merge form data and JSON body (for AJAX)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # 2. Extract and parse latitude/longitude
        lat = float(data.get('latitude', 0))
        lng = float(data.get('longitude', 0))

        # 3. Compute the single feature: distance_from_center
        distance = geodesic((lat, lng), DEFAULT_CITY_CENTER).km

        # 4. Predict using the one‑column model
        raw_pred = model.predict([[distance]])  # 2D list for sklearn
        price = round(float(raw_pred[0]), 2)

        # 5. Return JSON or HTML response
        if request.is_json:
            return jsonify(success=True, predicted_price=price)
        return render_template('result.html',
                               predicted_price=price,
                               input_data=data)

    except Exception as e:
        # Handle errors for both JSON and form
        err = str(e)
        if request.is_json:
            return jsonify(success=False, error=err), 400
        return render_template('result.html',
                               error=err,
                               input_data=data), 400

if __name__ == '__main__':
    app.run(debug=True)
