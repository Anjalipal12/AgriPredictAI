from flask import Flask, request, render_template
import numpy as np
import pickle

# Importing model and scalers
model = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Convert form inputs to float
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Prepare features and apply MinMaxScaler
    feature_list = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    final_features = ms.transform(feature_list)
    
    # Predict crop
    prediction = model.predict(final_features)

    # Crop mapping
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
        21: "Chickpea", 22: "Coffee"
    }

    # Generate result
    result = f"{crop_dict.get(prediction[0], 'Unknown')} is the best crop to be cultivated right now."
    
    return render_template('index.html', result=result)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run on port 5001

