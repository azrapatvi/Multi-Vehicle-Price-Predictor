# 🚗🏍️ Multi-Vehicle Price Predictor

A machine learning–based web application that predicts **Car** and **Bike** prices based on user inputs.  
The project includes **separate Exploratory Data Analysis (EDA)** and **separate trained models** for cars and bikes, all integrated into a single **Streamlit web app**.

---

## 📌 Project Overview

This project is designed to estimate the resale price of vehicles using historical data and machine learning models.

Key highlights:
- Separate **Car Price Prediction** and **Bike Price Prediction** pipelines
- Independent **EDA notebooks** for car and bike datasets
- Individually trained and saved models
- Unified **Streamlit application** for real-time predictions

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed **separately** for each vehicle type to understand pricing patterns and feature relationships.

### 🚗 Car Price EDA
- Analysis of vehicle age, mileage, engine capacity, power, fuel type, transmission, and seating capacity
- Impact of brand and seller type on car prices
- Outlier detection and distribution analysis

📓 Notebook: `car_price_prediction.ipynb`

---

### 🏍️ Bike Price EDA
- Relationship between bike price and age, kilometers driven, engine power, and ownership
- Brand-wise and owner-wise price trends
- City-level influence using encoding techniques

📓 Notebook: `Bike_Price_Prediction.ipynb`

---

## 🧠 Model Training

- Separate machine learning models were trained for:
  - **Car Price Prediction**
  - **Bike Price Prediction**
- Feature preprocessing and encoding were handled independently for each dataset
- Trained models and preprocessors were saved and reused in the web application

---

## 🌐 Web Application

The Streamlit app allows users to:
- Select vehicle type (**Car or Bike**)
- Enter relevant vehicle details
- Get an estimated resale price instantly

All predictions are powered by the trained machine learning models.

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Pickle


---

## 🎯 Use Case

- Helps users estimate fair resale prices
- Useful for buyers, sellers, and dealers
- Demonstrates end-to-end ML workflow with deployment


## 📂 Project Structure

