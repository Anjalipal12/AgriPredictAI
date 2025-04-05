 HEAD
# KrishiSathi
A complete Agriculture Solution

# AgriPredictAI
AgriPredictAI is an AI-powered farming system that predicts crop yield, recommends crops, detects plant diseases, and suggests fertilizers. It uses machine learning to analyze weather, soil, and historical data, helping farmers optimize resources, boost productivity, and make informed decisions.
 4da2cc9814919a75014c4adbd3dc819531c696e9

 Crop Recommendation System Using Machine Learning
The Crop Recommendation System is a machine learning-based application that provides crop recommendations based on various environmental and soil conditions. The primary goal is to assist farmers and agricultural professionals in making informed decisions, optimizing crop yields, and maximizing profitability.

By considering factors such as soil type, climate, rainfall, temperature, humidity, and pH levels, the system predicts the most suitable crops for specific regions. It leverages historical data and advanced predictive models to offer personalized recommendations tailored to the specific conditions of a farm or agricultural area.

Dataset
The dataset used in this project is built by augmenting rainfall, climate, and fertilizer data specific to India. The following attributes are included in the dataset:

N: Nitrogen content in the soil
P: Phosphorous content in the soil
K: Potassium content in the soil
Temperature: Temperature in degrees Celsius
Humidity: Relative humidity in %
pH: pH value of the soil
Rainfall: Rainfall in mm
Key Features
Input Data Collection: Allows users to input data such as soil parameters, climate information, and geographic location.

Data Preprocessing: Handles missing values, normalizes/scales features, and transforms categorical variables to prepare the data for analysis.

Machine Learning Models: Employs multiple machine learning algorithms, including Decision Trees, Random Forests, Support Vector Machines (SVM), and Gradient Boosting techniques, to build accurate predictive models.

Model Training and Evaluation: Models are trained on historical data and evaluated using performance metrics to ensure reliability and precision.

Crop Recommendation: Based on the trained models, the system recommends the most suitable crops for the given environmental and soil parameters.

Technologies Used
Python: Programming language used for model development, data preprocessing, and building the application.

Scikit-learn: Machine learning library used for training, evaluation, and making predictions.

Pandas: Data manipulation library used for preprocessing and analyzing the data.

NumPy: Numerical computing library used for handling arrays and performing mathematical operations.

Experiment Results
Data Analysis:
Most columns contain outliers, except for Nitrogen (N).
Performance Evaluation:
The dataset was split into 80% training data and 20% validation data.
Training and Validation:
Gaussian Naive Bayes (GaussianNB) outperformed other classification models.
GaussianNB ( 93.26 % accuracy score )
Performance Results
Training Accuracy: 93.26%
Validation Accuracy: 92.53%
