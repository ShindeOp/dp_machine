import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="🎓 Student Dropout Predictor", layout="wide")
st.title("🎓 Student Dropout Prediction App")
st.info("Upload the original student dataset and predict if a student will Dropout, Continue, or Graduate.")

# -------------------------------
# Dataset Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Original Dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=";")
        st.success("✅ Dataset loaded successfully!")

        target_col = "Target"
        if target_col not in df.columns:
            st.error("❌ 'Target' column not found in dataset.")
            st.stop()

        # Select important academic + personal features
        key_features = [
            "Admission grade",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 2nd sem (evaluations)",
            "Age at enrollment",
            "Tuition fees up to date",
            "Scholarship holder",
            "Course",
            "Gender",
            "Marital status",
            "Target"
        ]

        df = df[[col for col in key_features if col in df.columns]]

        with st.expander("📊 Preview Dataset"):
            st.dataframe(df.head())

        # ---------------------------------------
        # Encode and Train Model
        # ---------------------------------------
        X = df.drop("Target", axis=1).copy()
        y = df["Target"]

        label_encoders = {}
        for col in X.select_dtypes(include="object").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"✅ Model trained successfully (Accuracy: **{acc:.2f}**)")

        # ---------------------------------------
        # Input UI for Prediction
        # ---------------------------------------
        st.markdown("### 🧠 Enter Student Details for Prediction")

        col1, col2 = st.columns(2)

        # --- Academic Inputs ---
        with col1:
            st.subheader("📘 Academic Information")
            admission_grade = st.slider("Admission Grade (0 - 100)", 0, 100, 70)
            sem1_grade = st.slider("1st Sem Grade (0 - 20)", 0, 20, 12)
            sem2_grade = st.slider("2nd Sem Grade (0 - 20)", 0, 20, 12)
            sem1_approved = st.slider("1st Sem Subjects Approved (0 - 10)", 0, 10, 8)
            sem2_approved = st.slider("2nd Sem Subjects Approved (0 - 10)", 0, 10, 8)
            sem1_eval = st.slider("1st Sem Evaluations (0 - 10)", 0, 10, 6)
            sem2_eval = st.slider("2nd Sem Evaluations (0 - 10)", 0, 10, 6)

        # --- Personal Inputs ---
        with col2:
            st.subheader("👤 Personal Information")
            age = st.slider("Age at Enrollment", 17, 60, 22)
            tuition_status = st.selectbox("Tuition Fees Up to Date", ["Yes", "No"])
            scholarship = st.selectbox("Scholarship Holder", ["Yes", "No"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            course = st.text_input("Course Name", "Informatics")

        # ---------------------------------------
        # Prepare Sample
        # ---------------------------------------
        sample = {
            "Admission grade": admission_grade,
            "Curricular units 1st sem (grade)": sem1_grade,
            "Curricular units 2nd sem (grade)": sem2_grade,
            "Curricular units 1st sem (approved)": sem1_approved,
            "Curricular units 2nd sem (approved)": sem2_approved,
            "Curricular units 1st sem (evaluations)": sem1_eval,
            "Curricular units 2nd sem (evaluations)": sem2_eval,
            "Age at enrollment": age,
            "Tuition fees up to date": tuition_status,
            "Scholarship holder": scholarship,
            "Gender": gender,
            "Marital status": marital,
            "Course": course
        }

        # Encode categorical inputs
        for col, le in label_encoders.items():
            if col in sample:
                val = sample[col]
                if val not in le.classes_:
                    le.classes_ = np.append(le.classes_, val)
                sample[col] = le.transform([val])[0]

        sample_df = pd.DataFrame([sample])
        sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

        if st.button("🚀 Predict Outcome", use_container_width=True):
            pred_encoded = model.predict(sample_df)[0]
            predicted_label = target_encoder.inverse_transform([pred_encoded])[0]
            proba = model.predict_proba(sample_df).max() * 100

            st.subheader("📈 Prediction Result")
            if "Dropout" in predicted_label:
                st.error(f"❌ Predicted Outcome: **{predicted_label}** ({proba:.1f}% confidence)")
            elif "Graduate" in predicted_label:
                st.success(f"🎉 Predicted Outcome: **{predicted_label}** ({proba:.1f}% confidence)")
            else:
                st.info(f"➡️ Predicted Outcome: **{predicted_label}** ({proba:.1f}% confidence)")

    except Exception as e:
        st.error("⚠️ Error processing dataset.")
        st.exception(e)
else:
    st.warning("📤 Please upload your dataset to begin.")
