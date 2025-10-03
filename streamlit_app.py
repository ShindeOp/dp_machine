import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Streamlit page settings
st.set_page_config(page_title="🎓 Student Dropout Predictor", layout="wide")

st.title('🎓 Student Dropout Prediction App')
st.info('This app predicts whether a student will Dropout, Graduate, or remain Enrolled using a Random Forest model.')

# Load dataset
csv_url = "https://raw.githubusercontent.com/ASIF-Kh/Student-Dropout-Prediction/main/data.csv"

prediction_output_container = st.empty()

try:
    df = pd.read_csv(csv_url, sep=";")
    st.success("✅ Dataset loaded successfully!")

    with st.expander("📂 Preview Data"):
        st.dataframe(df.head(10))

    # --- Data Preprocessing ---
    y_original = df["Target"]
    X_original = df.drop("Target", axis=1)

    feature_encoders = {}
    X_encoded = X_original.copy()

    categorical_cols = X_original.select_dtypes(include='object').columns
    discrete_numeric_cols = [
        col for col in X_original.columns 
        if X_original[col].dtype != 'object' and 
           not np.issubdtype(X_original[col].dtype, np.floating) and 
           X_original[col].nunique() < 50
    ]
    all_categorical_cols = list(set(list(categorical_cols) + discrete_numeric_cols))
    
    for col in all_categorical_cols:
        X_original[col] = X_original[col].astype(str)
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_original[col])
        feature_encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_original)
    target_labels = dict(zip(target_encoder.transform(target_encoder.classes_), target_encoder.classes_))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation for more robust accuracy
    cv_scores = cross_val_score(model, X_encoded, y_encoded, cv=5)
    cv_mean = cv_scores.mean()

    st.success(f"✅ Random Forest trained successfully! Test Accuracy: **{acc:.2f}** | CV Accuracy: **{cv_mean:.2f}**")

    # Classification Report
    with st.expander("📊 Classification Report"):
        report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
        st.text(report)

    # Feature Importance
    with st.expander("🌟 Feature Importance"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importances)), importances[indices], align="center")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(X_encoded.columns[indices], rotation=90)
        ax.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig)

    # --- Prediction Input ---
    st.subheader("🎯 Try Prediction with Custom Input")

    sample_encoded = {}
    for col in X_original.columns:
        if col in feature_encoders:
            le = featur
