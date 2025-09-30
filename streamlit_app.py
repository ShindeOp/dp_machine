import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Set Streamlit page configuration (optional but good practice)
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

st.title('🎓 Student Dropout Prediction App')
st.info('This app predicts whether a student will Dropout, Graduate, or remain Enrolled.')

# Load dataset: UPDATED URL to the one provided in the ASIF-Kh repository
csv_url = "https://raw.githubusercontent.com/ASIF-Kh/Student-Dropout-Prediction/main/data.csv"

# Placeholder for prediction results, which we will populate later
prediction_output_container = st.empty()

try:
    # Load CSV with correct separator (The data.csv file from this repo is SEMICOLON-separated)
    df = pd.read_csv(csv_url, sep=";")
    st.success("Dataset loaded successfully from the new repository!")

    # Show first 10 rows
    with st.expander("📂 Preview Data"):
        st.dataframe(df.head(10))

    # --- Data Preprocessing: Store Encoders for better UI experience ---
    
    # 1. Separate Target and Features
    y_original = df["Target"]
    X_original = df.drop("Target", axis=1)

    # 2. Setup Encoders dictionary
    feature_encoders = {}
    X_encoded = X_original.copy()

    # Identify and encode categorical features (original 'object' dtypes)
    categorical_cols = X_original.select_dtypes(include='object').columns

    # Additionally, identify discrete numerical columns that should be treated as categories
    # E.g., 'Marital status', 'Course', 'Application mode'
    # We define a numerical column as categorical if it's not a float and has fewer than 50 unique values.
    discrete_numeric_cols = [
        col for col in X_original.columns 
        if X_original[col].dtype != 'object' and 
           not np.issubdtype(X_original[col].dtype, np.floating) and 
           X_original[col].nunique() < 50
    ]
    
    # Combine original categorical columns and the new discrete numerical ones
    all_categorical_cols = list(set(list(categorical_cols) + discrete_numeric_cols))
    
    for col in all_categorical_cols:
        # We need to ensure that the column data is consistent before encoding
        X_original[col] = X_original[col].astype(str) 
        
        le = LabelEncoder()
        # Fit and transform the column, storing the encoder
        X_encoded[col] = le.fit_transform(X_original[col])
        feature_encoders[col] = le
        
    # Encode Target (y)
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_original)
    
    # Create a mapping for decoding the final prediction (0, 1, 2) back to labels
    target_labels = dict(zip(target_encoder.transform(target_encoder.classes_), target_encoder.classes_))
    
    # 3. Split data (use encoded data)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

    # 4. Train model
    # Increased max_iter to ensure convergence for a large dataset
    model = LogisticRegression(max_iter=5000) 
    model.fit(X_train, y_train)

    # 5. Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained successfully! Test Accuracy: **{acc:.2f}**")

    # 6. Prediction demo with user input
    with st.expander("🎯 Try Prediction (Input Features)", expanded=True):
        st.write("Adjust the features below to get a prediction for a potential student.")
        
        # Dictionary to hold the final ENCODED feature values for prediction
        sample_encoded = {}
        
        # Use two columns for better layout
        cols = st.columns(2)
        col_index = 0

        for col in X_original.columns:
            with cols[col_index % 2]:
                
                # Check if the column was treated as categorical (it's in the encoders dictionary)
                if col in feature_encoders:
                    # Categorical feature: use selectbox
                    le = feature_encoders[col]
                    original_options = le.classes_
                    
                    # Set default index to the mode (most frequent)
                    # Use .iloc[0] for safety if mode returns multiple values
                    default_index = list(original_options).index(str(X_original[col].mode().iloc[0]))
                    
                    selected_original_val = st.selectbox(
                        f"{col}", 
                        options=original_options, 
                        index=default_index,
                        key=f"sb_{col}"
                    )
                    
                    # Manually transform the selected original value into the encoded integer
                    encoded_val = le.transform([selected_original_val])[0]
                    sample_encoded[col] = encoded_val
                    
                else:
                    # TRUE Numerical feature: use number input
                    data_col = X_original[col]
                    min_val = float(data_col.min())
                    max_val = float(data_col.max())
                    
                    is_float = np.issubdtype(data_col.dtype, np.floating)

                    if is_float:
                        # Handle continuous float columns
                        mean_val = float(data_col.mean())
                        step = 0.1 
                        format_str = "%.2f"
                    else:
                        # Handle true discrete integer columns (e.g., counters)
                        # We cast the mean to an int for the default value to match the format.
                        mean_val = float(int(data_col.mean())) 
                        step = 1.0 
                        format_str = "%d"
                    
                    val = st.number_input(
                        f"{col}", 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=mean_val, 
                        step=step, 
                        format=format_str,
                        key=f"ni_{col}"
                    )
                    sample_encoded[col] = val
            
            col_index += 1

        st.markdown("---")
        
        # Add a placeholder for the button inside the expander
        button_clicked = st.button("PREDICT STUDENT OUTCOME", type="primary", use_container_width=True)
    
    # 7. Prediction Output Logic (Moved outside the expander for visibility)
    if button_clicked:
        # Convert the encoded sample dictionary into a DataFrame for the model
        sample_df = pd.DataFrame([sample_encoded])
        
        # Ensure column order matches training data columns
        sample_df = sample_df[X_encoded.columns]
        
        # Make prediction
        pred_encoded = model.predict(sample_df)[0]
        
        # Decode the prediction for user-friendly output
        predicted_label = target_labels.get(pred_encoded, "Unknown Outcome")
        
        # --- Simplified Dropout/Not Dropout output ---
        is_dropout = 'Dropout' in predicted_label
        
        with prediction_output_container.container():
            st.subheader("Prediction Results:")
            st.markdown("---")
            
            # Display Dropout Status
            st.subheader("Dropout Status:")
            if is_dropout:
                st.error("❌ **PREDICTED DROPOUT**")
            else:
                st.success("✅ **NOT PREDICTED DROPOUT**")

            # Display result with styling (original detailed output)
            st.subheader("Detailed Prediction:")
            if is_dropout:
                 st.error(f"⚠️ **Predicted Outcome: {predicted_label}**")
                 st.caption("A student with these characteristics is highly likely to drop out. Consider intervention.")
            elif 'Graduate' in predicted_label:
                 st.balloons()
                 st.success(f"🎉 **Predicted Outcome: {predicted_label}**")
                 st.caption("A student with these characteristics is likely to graduate.")
            else: # Likely 'Enrolled'
                 st.info(f"➡️ **Predicted Outcome: {predicted_label}**")
                 st.caption("A student with these characteristics is predicted to remain enrolled.")


except Exception as e:
    # Clear the prediction container if there's an error during data load/training
    prediction_output_container.empty()
    st.error(f"An error occurred while loading data or training the model.")
    st.caption("Please ensure the CSV URL is correct and accessible.")
    st.exception(e)
