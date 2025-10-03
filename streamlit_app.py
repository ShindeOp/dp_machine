import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

st.title("🎓 Student Dropout Prediction App")
st.info("This app predicts whether a student will Dropout, Graduate, or remain Enrolled.")

# Dataset URL
csv_url = "https://raw.githubusercontent.com/ASIF-Kh/Student-Dropout-Prediction/main/data.csv"

# Placeholder for prediction results
prediction_output_container = st.empty()

try:
    # ----------------------------
    # Load Dataset
    # ----------------------------
    df = pd.read_csv(csv_url, sep=";")
    st.success("✅ Dataset loaded successfully!")

    # Show complete dataset preview
    with st.expander("📂 Preview Dataset"):
        st.dataframe(df)

    # ----------------------------
    # Data Preprocessing
    # ----------------------------
    y_original = df["Target"]
    X_original = df.drop("Target", axis=1)

    feature_encoders = {}
    X_encoded = X_original.copy()

    # Categorical columns
    categorical_cols = X_original.select_dtypes(include="object").columns

    # Discrete numeric columns (like IDs, categories encoded as numbers)
    discrete_numeric_cols = [
        col for col in X_original.columns
        if X_origi_
