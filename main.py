from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/crop_recommendation')
def crop_recommendation():
    return redirect('http://localhost:5001')  # Replace with the correct URL or port for your crop recommendation app

@app.route('/plant_disease_detection')
def plant_disease_detection():
    return redirect('http://localhost:5002')  # Replace with the correct URL or port for your plant disease detection app

@app.route('/fertilizer_recommendation')
def fertilizer_recommendation():
    return redirect('http://localhost:5003')  # Replace with the correct URL or port for your fertilizer recommendation app

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Main Flask application running on port 5000
