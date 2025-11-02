import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------
# Load trained model safely
# ---------------------------
model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model file 'trained_model.pkl' not found. Please upload it to the same directory.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Category Mappings
# ---------------------------

marital_status_map = {
    1: "Single",
    2: "Married",
    3: "Widow/Widower"
}

application_mode_map = {
    1: "Online",
    2: "Offline",
    3: "Referral"
}

daytime_evening_map = {
    1: "Daytime",
    2: "Evening"
}

previous_qualification_map = {
    1: "High School",
    2: "Bachelor’s Degree",
    3: "Master’s Degree",
    4: "Other"
}

nationality_map = {
    1: "Indian",
    2: "Foreign"
}

parents_qualification_map = {
    1: "No formal education",
    2: "High School",
    3: "Bachelor’s Degree",
    4: "Master’s Degree",
    5: "PhD"
}

parents_occupation_map = {
    1: "Unemployed",
    2: "Employed",
    3: "Self-employed",
    4: "Retired"
}

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🎓 Student Success Prediction App")

col1, col2 = st.columns(2)

with col1:
    marital_status_label = st.selectbox("Marital Status", marital_status_map.values())
    application_order = st.number_input("Application Order", min_value=1, value=1)
    daytime_label = st.selectbox("Daytime/Evening Attendance", daytime_evening_map.values())
    prev_grade = st.number_input("Previous Qualification (Grade)", min_value=0.0, value=100.0)
    mother_qual_label = st.selectbox("Mother's Qualification", parents_qualification_map.values())
    mother_occ_label = st.selectbox("Mother's Occupation", parents_occupation_map.values())

with col2:
    application_mode_label = st.selectbox("Application Mode", application_mode_map.values())
    course = st.number_input("Course", min_value=1000, value=9500)
    prev_qual_label = st.selectbox("Previous Qualification", previous_qualification_map.values())
    nationality_label = st.selectbox("Nationality", nationality_map.values())
    father_qual_label = st.selectbox("Father's Qualification", parents_qualification_map.values())
    father_occ_label = st.selectbox("Father's Occupation", parents_occupation_map.values())

# ---------------------------
# Convert labels back to numeric
# ---------------------------

marital_status = [k for k, v in marital_status_map.items() if v == marital_status_label][0]
application_mode = [k for k, v in application_mode_map.items() if v == application_mode_label][0]
daytime_evening = [k for k, v in daytime_evening_map.items() if v == daytime_label][0]
prev_qual = [k for k, v in previous_qualification_map.items() if v == prev_qual_label][0]
nationality = [k for k, v in nationality_map.items() if v == nationality_label][0]
mother_qual = [k for k, v in parents_qualification_map.items() if v == mother_qual_label][0]
father_qual = [k for k, v in parents_qualification_map.items() if v == father_qual_label][0]
mother_occ = [k for k, v in parents_occupation_map.items() if v == mother_occ_label][0]
father_occ = [k for k, v in parents_occupation_map.items() if v == father_occ_label][0]

# ---------------------------
# Prediction
# ---------------------------

features = np.array([[marital_status, application_mode, application_order, course,
                      daytime_evening, prev_qual, prev_grade, nationality,
                      mother_qual, father_qual, mother_occ, father_occ]])

if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ The student is likely to succeed!")
    else:
        st.warning("⚠️ The student may face challenges.")

