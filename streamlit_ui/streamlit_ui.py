import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("../ml_model/decision_tree_model.pkl")

# Get all feature names from your cleaned dataset
# You should add all column names from your cleaned_data.csv
all_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_0', 'sex_1', 
                'cp_0', 'cp_1', 'cp_2', 'cp_3', 'fbs_0', 'fbs_1', 
                'restecg_0.0', 'restecg_1.0', 'restecg_2.0', 'exang_0', 'exang_1', 
                'slope_0', 'slope_1', 'slope_2', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4', 
                'thal_0', 'thal_1', 'thal_2', 'thal_3']

# Streamlit app
st.title("Heart Disease Risk Prediction")

# Numerical inputs
st.header("Numerical Features")
age = st.slider("Age", 0, 100, 50)
trestbps = st.slider("Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)

# Categorical inputs
st.header("Categorical Features")
col1, col2 = st.columns(2)

with col1:
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", 
                     ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", 
                          ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])

with col2:
    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
    slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                        ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"])

if st.button("Predict Risk"):
    # Create a dictionary with all features initialized to 0
    input_dict = {feature: 0 for feature in all_features}
    
    # Fill in numerical features
    input_dict['age'] = age / 100  # Assuming normalization as in data_processing.ipynb
    input_dict['trestbps'] = (trestbps - 80) / (200 - 80)  # Simple min-max scaling
    input_dict['chol'] = (chol - 100) / (400 - 100)  # Simple min-max scaling
    input_dict['thalach'] = (thalach - 60) / (220 - 60)  # Simple min-max scaling
    input_dict['oldpeak'] = oldpeak / 6.0  # Simple min-max scaling
    
    # Fill in categorical features (one-hot encoded)
    # Sex
    if sex == "Male":
        input_dict['sex_1'] = 1
        input_dict['sex_0'] = 0
    else:
        input_dict['sex_0'] = 1
        input_dict['sex_1'] = 0
    
    # Chest Pain Type
    cp_mapping = {"Typical Angina": 'cp_0', "Atypical Angina": 'cp_1', 
                 "Non-anginal Pain": 'cp_2', "Asymptomatic": 'cp_3'}
    input_dict[cp_mapping[cp]] = 1
    
    # Fasting Blood Sugar
    if fbs == "Yes":
        input_dict['fbs_1'] = 1
        input_dict['fbs_0'] = 0
    else:
        input_dict['fbs_0'] = 1
        input_dict['fbs_1'] = 0
    
    # Resting ECG
    restecg_mapping = {"Normal": 'restecg_0.0', "ST-T Wave Abnormality": 'restecg_1.0', 
                      "Left Ventricular Hypertrophy": 'restecg_2.0'}
    input_dict[restecg_mapping[restecg]] = 1
    
    # Exercise Induced Angina
    if exang == "Yes":
        input_dict['exang_1'] = 1
        input_dict['exang_0'] = 0
    else:
        input_dict['exang_0'] = 1
        input_dict['exang_1'] = 0
    
    # Slope
    slope_mapping = {"Upsloping": 'slope_0', "Flat": 'slope_1', "Downsloping": 'slope_2'}
    input_dict[slope_mapping[slope]] = 1
    
    # Number of Major Vessels
    ca_feature = f'ca_{ca}'
    if ca_feature in input_dict:
        input_dict[ca_feature] = 1
    
    # Thalassemia
    thal_mapping = {"Normal": 'thal_0', "Fixed Defect": 'thal_1', 
                   "Reversible Defect": 'thal_2', "Unknown": 'thal_3'}
    input_dict[thal_mapping[thal]] = 1
    
    # Create DataFrame with proper order of features
    input_df = pd.DataFrame([input_dict])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    # Display result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease (Probability: {probability:.2f})")
    
    # Display input summary
    st.subheader("Summary of Inputs")
    st.write(f"Age: {age} years")
    st.write(f"Blood Pressure: {trestbps} mm Hg")
    st.write(f"Cholesterol: {chol} mg/dl")
    st.write(f"Maximum Heart Rate: {thalach}")
    st.write(f"ST Depression: {oldpeak}")
    st.write(f"Sex: {sex}")
    st.write(f"Chest Pain Type: {cp}")
    st.write(f"Fasting Blood Sugar > 120 mg/dl: {fbs}")
    st.write(f"Resting ECG Results: {restecg}")
    st.write(f"Exercise Induced Angina: {exang}")
    st.write(f"Slope of Peak Exercise ST Segment: {slope}")
    st.write(f"Number of Major Vessels Colored by Fluoroscopy: {ca}")
    st.write(f"Thalassemia: {thal}")