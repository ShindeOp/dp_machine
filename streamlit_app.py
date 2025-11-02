import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="🎓 Student Dropout Predictor (Kaggle Dataset)", layout="wide")
st.title("🎓 Student Dropout Prediction App (Kaggle Dataset)")
st.info("Predict whether a student will Dropout, Graduate, or remain Enrolled using key academic and demographic features.")

prediction_output_container = st.empty()

# -------------------------------------------------------
# Load Kaggle dataset (public mirror)
# -------------------------------------------------------
csv_url = "https://raw.githubusercontent.com/ASIF-Kh/Student-Dropout-Prediction/main/data.csv"

try:
    df = pd.read_csv(csv_url, sep=";")
    st.success("✅ Kaggle dataset loaded successfully!")

    # -------------------------------------------------------
    # Feature selection (top academic + personal + financial)
    # -------------------------------------------------------
    key_features = [
        "Age at enrollment", "Admission grade", "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)", "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)", "Application mode", "Course",
        "Scholarship holder", "Tuition fees up to date", "Mother's qualification",
        "Father's qualification", "Mother's occupation", "Father's occupation", "Target"
    ]
    df = df[key_features]

    # Normalize admission grade to 0–100
    if "Admission grade" in df.columns:
        df["Admission grade"] = (df["Admission grade"] / df["Admission grade"].max()) * 100

    # -------------------------------------------------------
    # Map categorical features to readable names
    # -------------------------------------------------------
    mappings = {
        "Application mode": {
            1: "1st Phase Contingent", 2: "Ordinance", 5: "International",
            6: "Other", 9: "Direct", 10: "2nd Phase Contingent", 12: "3rd Phase Contingent"
        },
        "Mother's qualification": {1: "Basic", 2: "Secondary", 3: "Graduate", 4: "Postgraduate"},
        "Father's qualification": {1: "Basic", 2: "Secondary", 3: "Graduate", 4: "Postgraduate"},
        "Mother's occupation": {1: "Unemployed", 2: "Employed", 3: "Self-Employed", 4: "Retired"},
        "Father's occupation": {1: "Unemployed", 2: "Employed", 3: "Self-Employed", 4: "Retired"},
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    with st.expander("📂 Preview Kaggle Dataset"):
        st.dataframe(df.head())

    # -------------------------------------------------------
    # Encode and prepare data
    # -------------------------------------------------------
    y = df["Target"]
    X = df.drop("Target", axis=1)
    X_encoded = X.copy()
    encoders = {}

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    target_labels = dict(zip(le_target.transform(le_target.classes_), le_target.classes_))

    # -------------------------------------------------------
    # Train model
    # -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"✅ Model trained successfully (Accuracy: **{acc:.2f}**)")

    # -------------------------------------------------------
    # Prediction Input UI
    # -------------------------------------------------------
    with st.expander("🎯 Predict a Student Outcome", expanded=True):
        sample = {}
        cols = st.columns(2)

        for i, col in enumerate(X.columns):
            with cols[i % 2]:
                if col in encoders:
                    val = st.selectbox(col, list(X[col].unique()), key=col)
                    sample[col] = encoders[col].transform([val])[0]
                else:
                    val = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()), key=col)
                    sample[col] = val

        if st.button("🚀 Predict", use_container_width=True):
            sample_df = pd.DataFrame([sample])
            pred = model.predict(sample_df)[0]
            result = target_labels[pred]

            with prediction_output_container.container():
                st.subheader("📊 Prediction Result")
                if "Dropout" in result:
                    st.error(f"❌ **Predicted Outcome: {result}**")
                elif "Graduate" in result:
                    st.balloons()
                    st.success(f"🎉 **Predicted Outcome: {result}**")
                else:
                    st.info(f"➡️ **Predicted Outcome: {result}**")

except Exception as e:
    st.error("⚠️ Error loading or processing dataset.")
    st.exception(e)
