# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('f2.csv')

# Clean column names
data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'}, inplace=True)

# Check for missing values and unique values
data.info()
data.isna().sum()

# Encode categorical variables
label_encoder = LabelEncoder()

# Encode 'Soil_Type'
data['Soil_Type'] = label_encoder.fit_transform(data['Soil_Type'])

# Encode 'Crop_Type'
data['Crop_Type'] = label_encoder.fit_transform(data['Crop_Type'])

# Encode 'Fertilizer'
data['Fertilizer'] = label_encoder.fit_transform(data['Fertilizer'])

# Split the data into training and testing sets
X = data.drop('Fertilizer', axis=1)
y = data['Fertilizer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Normalize the data for SVM
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Model Training & Evaluation
models = [
    ('Decision Tree', DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)),
    ('Naive Bayes', GaussianNB()),
    ('SVM', SVC(kernel='poly', degree=3, C=1)),
    ('Logistic Regression', LogisticRegression(random_state=2)),
    ('Random Forest', RandomForestClassifier(n_estimators=20, random_state=0))
]

# Store accuracy results
acc = []
acc_train = []
model_names = []

for name, model in models:
    model.fit(X_train_norm, y_train)
    
    # Test accuracy
    y_pred = model.predict(X_test_norm)
    acc.append(accuracy_score(y_test, y_pred))
    
    # Training accuracy
    y_pred_train = model.predict(X_train_norm)
    acc_train.append(accuracy_score(y_train, y_pred_train))
    
    model_names.append(name)
    
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# Cross-validation scores
for name, model in models:
    cv_score = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation score of {name}: {cv_score.mean():.4f}")

# Visualization of accuracy comparison
plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=model_names, palette='dark')

# Save the best model (Random Forest) using pickle
best_model = RandomForestClassifier(n_estimators=20, random_state=0)
best_model.fit(X_train, y_train)
pickle.dump(best_model, open('classifier.pkl', 'wb'))

# Save the label encoder for Fertilizer
pickle.dump(label_encoder, open('fertilizer.pkl', 'wb'))

# Example prediction using the saved model
loaded_model = pickle.load(open('classifier.pkl', 'rb'))
sample_data = [[34, 67, 62, 0, 1, 7, 0, 30]]  # Example data
predicted_fertilizer = loaded_model.predict(sample_data)
print(f"Predicted Fertilizer: {predicted_fertilizer[0]}")

# Load the label encoder and get the original class for the predicted fertilizer
loaded_encoder = pickle.load(open('fertilizer.pkl', 'rb'))
fertilizer_class = loaded_encoder.classes_[predicted_fertilizer[0]]
print(f"Fertilizer Class: {fertilizer_class}")