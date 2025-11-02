import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="🎓 Student Dropout Predictor (Optimized)", layout="wide")
st.title("🎓 Student Dropout Prediction App — Optimized")
st.info("Predict whether a student will Dropout, Graduate, or remain Enrolled using the most important features.")

prediction_output_container = st.empty()

csv_url = "https://raw.githubusercontent.com/ASIF-Kh/Student-Dropout-Prediction/main/data.csv"

try:
    df = pd.read_csv(csv_url, sep=";")
    st.success("✅ Dataset loaded successfully!")

    # -------------------------------------------------------
    # Keep only top important features + target
    # -------------------------------------------------------
    key_features = [
        "Age at enrollment",
        "Admission grade",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Application mode",
        "Course",
        "Scholarship holder",
        "Tuition fees up to date",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Target"
    ]
    df = df[key_features]

    # -------------------------------------------------------
    # Normalize Admission grade to 0–100 scale
    # -------------------------------------------------------
    if "Admission grade" in df.columns:
        df["Admission grade"] = (df["Admission grade"] / df["Admission grade"].max()) * 100

    # -------------------------------------------------------
    # Apply readable mappings for categorical columns
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

    with st.expander("📂 Preview Data"):
        st.dataframe(df.head())

    # -------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------
    y_original = df["Target"]
    X_original = df.drop("Target", axis=1).copy()
    X_encoded = X_original.copy()
    feature_encoders = {}

    categorical_cols = X_original.select_dtypes(include="object").columns
    discrete_numeric_cols = [
        c for c in X_original.columns
        if X_original[c].dtype != "object" and X_original[c].nunique() < 50
    ]
    all_categorical_cols = list(set(categorical_cols) | set(discrete_numeric_cols))

    for col in all_categorical_cols:
        X_original[col] = X_original[col].astype(str)
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_original[col])
        feature_encoders[col] = le

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_original)
    target_labels = dict(zip(target_encoder.transform(target_encoder.classes_), target_encoder.classes_))

    # -------------------------------------------------------
    # Train model
    # -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"✅ Model trained successfully (Accuracy: **{acc:.2f}**)")

    # -------------------------------------------------------
    # Prediction UI
    # -------------------------------------------------------
    with st.expander("🎯 Try Prediction (Input Features)", expanded=True):
        st.write("Adjust the features below to get a prediction.")
        sample_encoded = {}
        cols = st.columns(2)
        col_index = 0

        for col in X_original.columns:
            with cols[col_index % 2]:
                display_name = col.replace("_", " ").title()
                le = feature_encoders.get(col)

                # Dropdown for categorical
                if le:
                    options = list(le.classes_)
                    if all(str(opt).isdigit() for opt in options):
                        options = [f"Category {opt}" for opt in options]

                    default_val = str(X_original[col].mode().iloc[0])
                    default_index = options.index(default_val) if default_val in options else 0

                    selected = st.selectbox(display_name, options, index=default_index, key=f"sb_{col}")
                    selected_clean = selected.replace("Category ", "")
                    sample_encoded[col] = int(le.transform([selected_clean])[0])

                # Numeric input for continuous features
                else:
                    data_col = X_original[col]
                    val = st.number_input(
                        display_name,
                        float(data_col.min()),
                        float(data_col.max()),
                        float(data_col.mean()),
                        key=f"ni_{col}"
                    )
                    if np.issubdtype(X_encoded[col].dtype, np.integer):
                        val = int(round(val))
                    sample_encoded[col] = val

            col_index += 1

        st.markdown("---")
        button_clicked = st.button("🚀 Predict Student Outcome", type="primary", use_container_width=True)

    # -------------------------------------------------------
    # Prediction Output
    # -------------------------------------------------------
    if button_clicked:
        sample_df = pd.DataFrame([sample_encoded]).reindex(columns=X_encoded.columns, fill_value=0)
        sample_df = sample_df.astype(X_encoded.dtypes.to_dict())

        pred_encoded = model.predict(sample_df)[0]
        predicted_label = target_labels.get(pred_encoded, "Unknown Outcome")

        with prediction_output_container.container():
            st.subheader("📊 Prediction Results")
            if "Dropout" in predicted_label:
                st.error(f"❌ **Predicted Outcome: {predicted_label}**")
            elif "Graduate" in predicted_label:
                st.balloons()
                st.success(f"🎉 **Predicted Outcome: {predicted_label}**")
            else:
                st.info(f"➡️ **Predicted Outcome: {predicted_label}**")

except Exception as e:
    st.error("⚠️ Error loading or processing dataset.")
    st.exception(e)
