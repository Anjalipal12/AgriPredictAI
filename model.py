import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
crop = pd.read_csv("Crop_recommendation.csv")

# Define crop mapping
drop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20, 
    'chickpea': 21, 'coffee': 22
}

# Encode crop labels
crop['crop_num'] = crop['label'].map(drop_dict)
crop.drop(['label'], axis=1, inplace=True)

# Split dataset
X = crop.drop(['crop_num'], axis=1)
y = crop['crop_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
ms = MinMaxScaler()
ss = StandardScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

# Train model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Save model and scalers
pickle.dump(rfc, open('model.pkl', 'wb'))
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))
pickle.dump(ss, open('standscalar.pkl', 'wb'))

print("Model training complete. Saved as model.pkl, minmaxscaler.pkl, and standscalar.pkl.")