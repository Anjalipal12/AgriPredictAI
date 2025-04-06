
# 🌾 KrishiSathi - AgriPredictAI

**AgriPredictAI** is a complete AI-powered agriculture solution designed to assist farmers in making smart, data-driven decisions. It offers:

- ✅ Crop Recommendation  
- 🧪 Fertilizer Suggestion  
- 🌿 Plant Disease Detection  

By analyzing environmental factors like weather, soil nutrients, temperature, and rainfall, this system enhances productivity and resource utilization for better farming outcomes.

---

## 📌 Project Highlights

### 🚜 Crop Recommendation System

A machine learning-based system that predicts the most suitable crop to grow based on:

- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- Soil pH
- Rainfall (mm)

**🔍 Goal:** Help farmers and agriculture professionals choose crops tailored to their local conditions, thereby increasing yield and profitability.

---

### 🧠 Technologies Used

- **Python** – Core programming
- **Scikit-learn** – ML models
- **Pandas & NumPy** – Data preprocessing & analysis
- **Flask** – Web deployment
- **PyTorch** – Deep Learning (for disease detection)

---

## 📊 Datasets Used

- Crop Recommendation Dataset (India-based)
- PlantVillage Leaf Image Dataset (for disease detection)

---

## 🧪 Machine Learning Models

### ✅ Crop Prediction:
- Algorithms used: Decision Tree, Random Forest, SVM, Gradient Boosting, GaussianNB
- **Best Performing:** Gaussian Naive Bayes (93.26% accuracy)

> 📈 **Training Accuracy**: 93.26%  
> 📉 **Validation Accuracy**: 92.53%

---

### 🌿 Plant Disease Detection

- Based on CNN (Convolutional Neural Network)
- Framework: **PyTorch**
- Dataset: **PlantVillage** (39 leaf classes)
- Output: Classifies plant leaf image into disease category

---

## 🖥️ How to Run Project Locally

### Prerequisites

- Python 3.8+
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Anjalipal12/AgriPredictAI.git

# 2. Move to the project folder
cd AgriPredictAI/AgriPredictAI

# 3. Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download the trained model file (plant_disease_model_1.pt) and place it in the root folder

# 6. Run the app
python app.py
```

> Open browser at `http://127.0.0.1:5000` to use the web interface.

---

## 🧪 Testing

If you don’t have your own leaf images, use the ones provided in `test_images/` folder.  
Each image has the corresponding disease name for comparison.

---

## 📸 Screenshots

You can add screenshots of your web interface here (e.g., crop prediction form, disease detection results, etc.)

![Homepage](static/images/home.png)  
![Prediction Result](static/images/result.png)

---

## 🤝 Contribution Guidelines

- Fork the project and create a new branch
- Add your features / fix bugs
- Ensure your code runs error-free
- Submit a pull request with a proper description

🛠️ You can contribute in:
- Improving UI
- Enhancing Deep Learning model
- Writing detailed markdown/docs
- Bug fixing

🔗 [How to Create Pull Request](https://opensource.com/article/19/7/create-pull-request-github)

---

## 📝 License

This project is open source and available under the **MIT License**.

---

🔗 **Plant Disease Detection Using CNN with PyTorch** *(Add blog URL here if available)*
