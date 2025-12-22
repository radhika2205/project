import streamlit as st
import pandas as pd
import joblib
import os

#  ALWAYS FIRST
st.set_page_config(page_title="Accident Severity Prediction")

#  Safe model loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), "accident_pipeline_model.pkl")
model = joblib.load(MODEL_PATH)

st.title("🚦 Smart Accident Severity Prediction")
st.write("Predict accident severity using Supervised ML")

# -------- INPUTS --------
light = st.selectbox("Light Conditions", ["Daylight", "Darkness"])
weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy"])
road = st.selectbox("Road Surface", ["Dry", "Wet"])
speed = st.slider("Vehicle Speed", 0, 150, 40)
alcohol = st.selectbox("Alcohol Involved", ["Yes", "No"])
seatbelt = st.selectbox("Seatbelt Used", ["Yes", "No"])

# Create dataframe ( SAME STRUCTURE AS TRAINING)
input_df = pd.DataFrame([{
    "Light_conditions": light,
    "Weather_conditions": weather,
    "Road_surface_conditions": road,
    "Vehicle_speed": speed,
    "Alcohol_involved": alcohol,
    "Seatbelt_used": seatbelt
}])

# Prediction
if st.button("Predict Severity"):
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success("🟢 Slight Injury")
    elif prediction == 1:
        st.warning("🟡 Serious Injury")
    else:
        st.error("🔴 Fatal Injury")
