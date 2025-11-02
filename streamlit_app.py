import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Outcome Predictor", layout="wide")
st.title("🎓 Student Performance & Dropout Prediction")

# ---------------------------------------
# Load Original Dataset Automatically
# ---------------------------------------
DATA_PATH = "student.csv"  # 👈 change filename if different
df = pd.read_csv(DATA_PATH)
st.success("✅ Original dataset loaded successfully!")

# Show dataset
with st.expander("🔍 Preview Original Dataset"):
    st.dataframe(df.head())

# ---------------------------------------
# Encode Data
# ---------------------------------------
target_col = df.columns[-1]  # Assuming last column is the target
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical features
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
st.info(f"✅ Model trained with accuracy: **{acc*100:.2f}%**")

st.divider()
st.subheader("📊 Enter Student Details for Prediction")

# ---------------------------------------
# Input Section
# ---------------------------------------
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧮 Academic Details")
        admission_grade = st.slider("Admission grade (0–100)", 0, 100, 70)
        sem1_grade = st.slider("1st Sem Grade (0–10)", 0.0, 10.0, 7.0)
        sem2_grade = st.slider("2nd Sem Grade (0–10)", 0.0, 10.0, 7.0)
        sem1_eval = st.number_input("1st Sem Evaluations", 0, 50, 10)
        sem2_eval = st.number_input("2nd Sem Evaluations", 0, 50, 10)
        sem1_approved = st.number_input("1st Sem Approved", 0, 20, 8)
        sem2_approved = st.number_input("2nd Sem Approved", 0, 20, 8)

    with col2:
        st.markdown("### 👤 Personal Details")
        age = st.slider("Age at Enrollment", 15, 60, 20)
        tuition = st.selectbox("Tuition Fees Up To Date", ["Yes", "No"])
        scholarship = st.selectbox("Scholarship Holder", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widower"])
        course = st.text_input("Course", "Computer Science")

    submitted = st.form_submit_button("🚀 Predict Outcome")

    if submitted:
        # Prepare Input
        sample = {
            "Admission grade": admission_grade,
            "Curricular units 1st sem (grade)": sem1_grade,
            "Curricular units 2nd sem (grade)": sem2_grade,
            "Curricular units 1st sem (approved)": sem1_approved,
            "Curricular units 2nd sem (approved)": sem2_approved,
            "Curricular units 1st sem (evaluations)": sem1_eval,
            "Curricular units 2nd sem (evaluations)": sem2_eval,
            "Age at enrollment": age,
            "Tuition fees up to date": tuition,
            "Scholarship holder": scholarship,
            "Gender": gender,
            "Marital status": marital,
            "Course": course
        }

        sample_df = pd.DataFrame([sample])

        # Encode categorical columns
        for col, le in label_encoders.items():
            if col in sample_df.columns:
                val = sample_df.at[0, col]
                if val not in le.classes_:
                    le.classes_ = np.append(le.classes_, val)
                sample_df[col] = le.transform([val])

        # Align columns
        sample_df = sample_df.reindex(columns=X.columns, fill_value=0)
        sample_df = sample_df.astype(float)

        # Predict
        pred_encoded = model.predict(sample_df)[0]
        predicted_label = target_encoder.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(sample_df).max() * 100

        st.divider()
        st.subheader("📈 Prediction Result")

        if "Dropout" in predicted_label:
            st.error(f"❌ Predicted Outcome: **{predicted_label}** ({proba:.1f}% confidence)")
        elif "Graduate" in predicted_label:
            st.success(f"
