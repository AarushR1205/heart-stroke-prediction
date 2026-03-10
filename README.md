# ❤️ Heart Stroke Predictor

A Machine Learning powered web application that predicts the **risk of heart stroke** based on patient health parameters.  
The application uses a trained **K-Nearest Neighbors (KNN)** model and provides an interactive **Streamlit dashboard** for predictions.

---

## 🚀 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help individuals take preventive measures and seek timely medical advice.

This project uses machine learning techniques to analyze medical features such as age, blood pressure, cholesterol level, ECG results, and heart rate to predict the likelihood of heart stroke.

The model is deployed using **Streamlit** to provide a simple and interactive user interface where users can input their health details and instantly get prediction results.

---

## 🧠 Machine Learning Model

Algorithm used:

- K-Nearest Neighbors (KNN)

Steps performed in the project:

1. Data preprocessing
2. Feature engineering
3. Model training
4. Model evaluation
5. Model deployment with Streamlit

The trained model is saved using **Joblib** and loaded inside the web application. 
Model Accuracy : 87%

---

## 📊 Features of the Application

✔ Modern Streamlit Dashboard UI  
✔ Real-time heart disease risk prediction  
✔ Risk probability visualization  
✔ Health recommendations based on prediction  
✔ Interactive input sliders and selection fields  
✔ Data preprocessing and scaling before prediction  

---

## 🏥 Input Features Used

The model uses the following medical parameters:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- ST Slope

---

## 🖥️ Tech Stack

**Programming Language**
- Python

**Libraries**
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Plotly
- Joblib

---

## 📂 Project Structure

```
Heart-Stroke-Predictor
│
├── app.py
├── knn_heart_model.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
└── README.md
```

---

## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/AarushR1205/heart-stroke-prediction.git
```

Navigate to project folder:

```bash
cd heart-stroke-prediction
```

Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib plotly
```

Run the application:

```bash
streamlit run app.py
```

Open the app in browser:

```
http://localhost:8501
```

---

## 📊 Example Workflow

1️⃣ Enter patient health details  
2️⃣ Click **Predict Heart Stroke Risk**  
3️⃣ The model processes the inputs  
4️⃣ Prediction and risk probability are displayed  

---

## 🧑‍💻 Author

**Aarush Rawat**

Aspiring AI / Machine Learning Engineer

GitHub:
https://github.com/AarushR1205

---

## ⭐ Future Improvements

- Explainable AI (SHAP)
- Multiple ML model comparison
- Model accuracy visualization
- Cloud deployment
- Real medical dataset integration

---

## 📜 License

This project is open-source and available under the **MIT License**.

## Project Demo
<img width="800" alt="image" src="https://github.com/user-attachments/assets/e0bf9bfe-86da-4f4a-a967-21352e0ac9cf" />
