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

⭐Plant-Disease-Detection
Plant Disease is necessary for every farmer so we are created Plant disease detection using Deep learning. In which we are using convolutional Neural Network for classifying Leaf images into 39 Different Categories. The Convolutional Neural Code build in Pytorch Framework. For Training we are using Plant village dataset. Dataset Link is in My Blog Section.
⭐Run Project in your Machine
You must have Python3.8 installed in your machine.
Create a Python Virtual Environment & Activate Virtual Environment Link
Install all the dependencies using below command pip install -r requirements.txt
Go to the Flask Deployed App folder.
Download the pre-trained model file plant_disease_model_1.pt from here
Add the downloaded file in Flask Deployed App folder.
Run the Flask app using below command python3 app.py
You can also use downloaded file in Model Section and play with it using Jupyter Notebook.
⭐Contribution ( Open Source )
This Project is now open source.
All the developers who are intrested they can contribute in this project.
Yo can make UI better , make Deep learning model more powerful , add informative markdown file in section...
If you will change Deep learning make sure you upload updated markdown file (.md) , .pdf and .ipynb in particular section.
Make sure your code is working. It will not have any type or error.
You have to fork this project then make a pull request after you testing will successful.
How to make pull request : https://opensource.com/article/19/7/create-pull-request-github
⭐Testing Images
If you do not have leaf images then you can use test images located in test_images folder
Each image has its corresponding disease name, so you can verify whether the model is working perfectly or not
⭐Blog Link
Plant Disease Detection Using Convolutional Neural Networks with PyTorch
