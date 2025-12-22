import streamlit as st
import pandas as pd
import os
import joblib

st.set_page_config(page_title="Accident Severity Prediction")

# ---------- SAFE MODEL LOADING ----------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "accident_pipeline_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ---------- UI ----------
st.title("🚦 Smart Accident Severity Prediction")
st.write("Predict accident severity using Supervised ML")

# -------- INPUTS --------
light = st.selectbox("Light Conditions", ["Daylight", "Darkness"])
weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy"])
road = st.selectbox("Road Surface", ["Dry", "Wet"])
speed = st.slider("Vehicle Speed (km/h)", 0, 150, 40)
alcohol = st.selectbox("Alcohol Involved", ["Yes", "No"])
seatbelt = st.selectbox("Seatbelt Used", ["Yes", "No"])

# ---------- INPUT DATAFRAME ----------
input_df = pd.DataFrame([{
    "Light_conditions": light,
    "Weather_conditions": weather,
    "Road_surface_conditions": road,
    "Vehicle_speed": speed,
    "Alcohol_involved": alcohol,
    "Seatbelt_used": seatbelt
}])

# ---------- PREDICTION ----------
if st.button("Predict Severity"):
    try:
        prediction = model.predict(input_df)[0]

        if prediction == 0:
            st.success("🟢 Slight Injury")
        elif prediction == 1:
            st.warning("🟡 Serious Injury")
        else:
            st.error("🔴 Fatal Injury")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.write(e)
