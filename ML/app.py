import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model & columns
model = joblib.load("xgboost_accident_model.pkl")
columns = joblib.load("xgboost_model_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🚗 Enhanced Accident Severity Prediction")
st.markdown("This enhanced model considers more factors for better predictions")

# ---------------- USER INPUTS ----------------
st.header("Driver Information")
col1, col2, col3 = st.columns(3)
with col1:
    age_band = st.selectbox("Age Band of Driver", ["Under 18", "18-30", "31-50", "Over 51"])
    sex = st.selectbox("Sex of Driver", ["Male", "Female"])
with col2:
    education = st.selectbox("Educational Level", ["Junior high school", "High school", "Above high school"])
    driving_exp = st.selectbox("Driving Experience", ["Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr"])
with col3:
    vehicle_relation = st.selectbox("Vehicle Driver Relation", ["Owner", "Employee", "Other"])
    service_year = st.selectbox("Service Year of Vehicle", ["Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "Unknown"])

st.header("Vehicle Information")
col1, col2, col3 = st.columns(3)
with col1:
    vehicle_type = st.selectbox("Type of Vehicle", ["Car", "Bus", "Truck", "Motorcycle"])
    owner = st.selectbox("Owner of Vehicle", ["Owner", "Governmental", "Organization", "Other"])
with col2:
    defect = st.selectbox("Defect of Vehicle", ["No defect", "Defective"])
    lanes = st.selectbox("Lanes or Medians", ["Single carriageway", "Double carriageway", "Undivided Two way", "Other"])
with col3:
    movement = st.selectbox("Vehicle Movement", ["Going straight", "Turning", "U-Turn", "Moving Backward", "Getting off", "Overtaking", "Other", "Parked"])

st.header("Environment Conditions")
col1, col2, col3 = st.columns(3)
with col1:
    time = st.selectbox("Time", ["Morning", "Day", "Evening", "Night"])
    day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
with col2:
    weather = st.selectbox("Weather Conditions", ["Clear", "Rainy", "Fog"])
    light = st.selectbox("Light Conditions", ["Daylight", "Darkness - lights lit", "Darkness - no lighting", "Darkness - lights unlit"])
with col3:
    road_surface = st.selectbox("Road Surface Conditions", ["Dry", "Wet", "Snow", "Ice", "Sand", "Wet or damp", "Other"])
    road_alignment = st.selectbox("Road Alignment", ["Straight road", "Curved diversion", "Two-way (divided with solid lines road marking)", "Other"])

st.header("Accident Details")
col1, col2, col3 = st.columns(3)
with col1:
    collision_type = st.selectbox("Type of Collision", ["Collision with roadside objects", "Collision with roadside-parked vehicles", "Vehicle with vehicle collision", "Rear collision", "Other"])
    junction = st.selectbox("Types of Junction", ["No junction", "Y Shape", "Crossing", "O Shape", "Other"])
with col2:
    area = st.selectbox("Area Accident Occurred", ["Other", "Office areas", "Residential areas", " Church areas", "Industrial areas", "School areas", "Hospital areas"])
    casualties = st.number_input("Number of Casualties", 1, 20, 1)
with col3:
    vehicles_involved = st.number_input("Number of Vehicles Involved", 1, 10, 1)
    st.markdown("\n\n")  # Empty space instead of speed limit

st.header("Casualty Information")
col1, col2, col3 = st.columns(3)
with col1:
    casualty_class = st.selectbox("Casualty Class", ["Driver", "Passenger", "Pedestrian"])
    casualty_sex = st.selectbox("Sex of Casualty", ["Male", "Female"])
with col2:
    casualty_age = st.selectbox("Age Band of Casualty", ["Under 18", "18-30", "31-50", "Over 51"])
    casualty_severity = st.selectbox("Casualty Severity", ["Minor", "Serious Injury", "Fatal injury"])
with col3:
    work = st.selectbox("Work of Casualty", ["Driver", "Businessman", "Student", "Employee", "Other"])
    fitness = st.selectbox("Fitness of Casualty", ["Normal", "Blind or vision defective", "Deaf or hearing defective"])

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Severity"):
    try:
        # 🔥 Step-1: Create a dictionary with all required columns
        # Initialize with empty strings for categorical and 0 for numerical
        # Numerical columns based on the model
        numerical_columns = ['Number_of_casualties', 'Number_of_vehicles_involved']
        input_dict = {}
        for col in columns:
            if col in numerical_columns:
                input_dict[col] = 0
            else:
                input_dict[col] = ''
        
        # 🔥 Step-2: Fill in the user inputs
        input_dict["Age_band_of_driver"] = age_band
        input_dict["Sex_of_driver"] = sex
        input_dict["Educational_level"] = education
        input_dict["Driving_experience"] = driving_exp
        input_dict["Vehicle_driver_relation"] = vehicle_relation
        input_dict["Service_year_of_vehicle"] = service_year
        input_dict["Type_of_vehicle"] = vehicle_type
        input_dict["Owner_of_vehicle"] = owner
        input_dict["Defect_of_vehicle"] = defect
        input_dict["Lanes_or_Medians"] = lanes
        input_dict["Vehicle_movement"] = movement
        input_dict["Time"] = time
        input_dict["Day_of_week"] = day
        input_dict["Weather_conditions"] = weather
        input_dict["Light_conditions"] = light
        input_dict["Road_surface_conditions"] = road_surface
        input_dict["Road_allignment"] = road_alignment
        input_dict["Type_of_collision"] = collision_type
        input_dict["Types_of_Junction"] = junction
        input_dict["Area_accident_occured"] = area
        input_dict["Number_of_casualties"] = int(casualties)
        input_dict["Number_of_vehicles_involved"] = int(vehicles_involved)
        input_dict["Casualty_class"] = casualty_class
        input_dict["Sex_of_casualty"] = casualty_sex
        input_dict["Age_band_of_casualty"] = casualty_age
        input_dict["Casualty_severity"] = casualty_severity
        input_dict["Work_of_casuality"] = work
        input_dict["Fitness_of_casuality"] = fitness
        
        # 🔥 Step-3: Create DataFrame
        input_data = pd.DataFrame([input_dict])
        
        # 🔥 Step-4: Predict
        prediction_encoded = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Decode prediction
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # 🔥 Step-5: Display results
        st.success(f"✅ Predicted Accident Severity: **{prediction}**")
        
        # Show probabilities
        st.subheader("Prediction Probabilities")
        prob_data = pd.DataFrame({
            'Severity': label_encoder.classes_,
            'Probability': [f"{prob:.2%}" for prob in probabilities]
        })
        st.table(prob_data)
        
        # Provide interpretation
        st.subheader("Interpretation")
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        
        if max_prob > 0.7:
            confidence = "High"
        elif max_prob > 0.5:
            confidence = "Moderate"
        else:
            confidence = "Low"
            
        st.info(f"Confidence Level: {confidence} ({max_prob:.1%})")
        
        # Risk assessment
        if prediction == "Fatal injury":
            st.error("🚨 HIGH RISK - Take extra precautions!")
        elif prediction == "Serious Injury":
            st.warning("⚠️ MODERATE RISK - Be careful!")
        else:
            st.success("🟢 LOW RISK - Standard caution advised")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Add information about the model
st.sidebar.header("ℹ️ About This Model")
st.sidebar.markdown("""
**Enhanced Accident Severity Prediction Model**

- Uses XGBoost algorithm
- Trained on 12,316 accident records
- Handles class imbalance with advanced techniques
- Considers 31 different factors
- Provides probability estimates
""")