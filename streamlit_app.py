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

# Load dataset
csv_url = "https://raw.githubusercontent.com/ShindeOp/dp_machine/master/data.csv"

try:
    # Load CSV with correct separator
    df = pd.read_csv(csv_url, sep=";")
    st.success("Dataset loaded successfully!")

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
    
    # Identify and encode categorical features
    categorical_cols = X_original.select_dtypes(include='object').columns
    
    for col in categorical_cols:
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
                
                # Check if the column was encoded (i.e., it's categorical)
                if col in feature_encoders:
                    # Categorical feature: use selectbox
                    le = feature_encoders[col]
                    original_options = le.classes_
                    
                    # Set default index to the mode (most frequent)
                    default_index = list(original_options).index(X_original[col].mode()[0])
                    
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
                    # Numerical feature: use number input
                    data_col = X_original[col]
                    min_val = float(data_col.min())
                    max_val = float(data_col.max())
                    mean_val = float(data_col.mean())
                    
                    # Adjust step and format for floats vs. integers
                    is_float = np.issubdtype(data_col.dtype, np.floating)
                    # FIX: Ensure step is always a float to avoid StreamlitMixedNumericTypesError
                    step = 0.1 if is_float else 1.0 
                    format_str = "%.2f" if is_float else "%d"
                    
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
        if st.button("PREDICT STUDENT OUTCOME", type="primary", use_container_width=True):
            # Convert the encoded sample dictionary into a DataFrame for the model
            sample_df = pd.DataFrame([sample_encoded])
            
            # Ensure column order matches training data columns
            sample_df = sample_df[X_encoded.columns]
            
            # Make prediction
            pred_encoded = model.predict(sample_df)[0]
            
            # Decode the prediction for user-friendly output
            predicted_label = target_labels.get(pred_encoded, "Unknown Outcome")
            
            # Display result with styling
            st.subheader("Prediction Result:")
            if 'Dropout' in predicted_label:
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
    st.error(f"An error occurred while loading data or training the model.")
    st.caption("Please ensure the CSV URL is correct and accessible.")
    st.exception(e)
