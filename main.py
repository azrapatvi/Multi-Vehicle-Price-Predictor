import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st

def indian(n):
    s = str(int(n))
    out = s[-3:]
    s = s[:-3]
    while len(s) > 0:
        out = s[-2:] + ',' + out
        s = s[:-2]
    return out
st.sidebar.title("Multi-Product Price Predictor")

product = st.sidebar.selectbox(
    "Select Product",
    ['Car', 'Bike']
)

if product == 'Car':
    st.title("🚗 Car Price Predictor")
    st.markdown("Enter the car details below:")

    with st.form(key='car_form'):
        brand = st.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Other'])
        model_name = st.text_input("Model Name")
        vehicle_age = st.number_input("Vehicle Age (years)", 0, 30, 5)
        km_driven = st.number_input("KM Driven", 0, 1000000, 20000)
        seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
        fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])
        transmission_type = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
        mileage = st.number_input("Mileage (km/l)", 0.0, 50.0, 18.0)
        engine = st.number_input("Engine (CC)", 0, 5000, 1200)
        max_power = st.number_input("Max Power (BHP)", 0.0, 1000.0, 80.0)
        seats = st.number_input("Seats", 2, 12, 5)

        predict_btn = st.form_submit_button("Predict Price")

    if predict_btn:
        new_df = pd.DataFrame({
            'brand': [brand],
            'model': [model_name],
            'vehicle_age': [vehicle_age],
            'km_driven': [km_driven],
            'seller_type': [seller_type],
            'fuel_type': [fuel_type],
            'transmission_type': [transmission_type],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats]
        })

        with open("car_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)

        with open("car_preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        with open("car_price_predictor_model.pkl", "rb") as f:
            model = pickle.load(f)

        for col in ['brand', 'model']:
            new_df[col] = new_df[col].apply(
                lambda x: encoders[col].transform([x])[0]
                if x in encoders[col].classes_
                else -1
            )

        new_df = preprocessor.transform(new_df)
        prediction = model.predict(new_df)

        st.success(f"💰 Estimated Car Price: ₹ {indian(prediction[0])}")
if product == 'Bike':
    st.title("🏍️ Bike Price Predictor")

    with st.form("bike_form"):
        kms_driven = st.number_input("Kilometers Driven", 0, 300000, 15000)

        owner = st.selectbox(
            "Owner Type",
            ["First Owner", "Second Owner", "Third Owner", "Fourth Owner"]
        )

        age = st.number_input("Bike Age (years)", 0, 30, 3)

        power = st.number_input("Engine Power (CC)", 50, 2000, 110)

        brand = st.selectbox(
            "Brand",
            ["TVS", "Royal Enfield", "Bajaj", "Honda", "Hero", "Yamaha"]
        )

        city = st.text_input("City")

        predict_btn = st.form_submit_button("Predict Price")

    if predict_btn:
        input_df = pd.DataFrame({
            "kms_driven": [kms_driven],
            "owner": [owner],
            "age": [age],
            "power": [power],
            "brand": [brand]
        })

        with open("bike_city_encoder.pkl", "rb") as f:
            city_encoder = pickle.load(f)

        with open("bike_preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        with open("bike_price_predictor_model.pkl", "rb") as f:
            model = pickle.load(f)

        # city encoding (target encoding)
        input_df["city_encoded"] = city_encoder.get(city, city_encoder.mean())

        # preprocess & predict
        input_df = preprocessor.transform(input_df)
        prediction = model.predict(input_df)

        st.success(f"💰 Estimated Bike Price: ₹ {indian(prediction[0])}")
