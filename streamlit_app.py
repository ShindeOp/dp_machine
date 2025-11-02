import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load saved model and encoders
model = pickle.load(open("trained_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
X_original = pickle.load(open("X_original.pkl", "rb"))

st.title("🎓 Student Success Prediction App")
st.write("Adjust the features below to predict the likelihood of student success.")

# Function to identify categorical vs numerical columns
def get_column_type(df, col):
    if df[col].dtype == 'object':
        return 'categorical'
    if len(df[col].unique()) < 10 and not np.issubdtype(df[col].dtype, np.floating):
        return 'categorical'
    return 'numerical'

sample_encoded = {}

cols = st.columns(2)
col_index = 0

for col in X_original.columns:
    col_type = get_column_type(X_original, col)

    # Handle categorical columns (with readable labels)
    if col in encoders and col_type == 'categorical':
        le = encoders[col]

        # Ensure labels are readable (not numeric)
        options = list(le.classes_)
        if all(str(opt).isdigit() for opt in options):
            # Map numeric-looking codes to readable placeholders
            options = [f"Category {opt}" for opt in options]

        default_index = 0
        try:
            default_index = options.index(str(X_original[col].mode().iloc[0]))
        except ValueError:
            pass

        with cols[col_index]:
            selected_display = st.selectbox(
                f"{col.replace('_', ' ').title()}",
                options,
                index=default_index,
                key=f"sb_{col}"
            )

        # Encode selected value
        sample_encoded[col] = le.transform(
            [selected_display.replace("Category ", "")]
        )[0]

    # Handle numerical columns
    else:
        default_val = float(X_original[col].mean())
        min_val = float(X_original[col].min())
        max_val = float(X_original[col].max())

        with cols[col_index]:
            sample_encoded[col] = st.number_input(
                f"{col.replace('_', ' ').title()}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"num_{col}"
            )

    col_index = (col_index + 1) % 2

# Convert sample to DataFrame
sample_df = pd.DataFrame([sample_encoded])

# Predict button
if st.button("Predict"):
    prediction = model.predict(sample_df)[0]
    st.subheader("🎯 Prediction Result:")
    st.write("✅ Student Likely to Succeed" if prediction == 1 else "⚠️ Student At Risk")
