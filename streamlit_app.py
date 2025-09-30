import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title('🎓 Student Dropout Prediction App')
st.info('This app predicts whether a student will Dropout or Graduate')

# Load dataset
csv_url = "https://raw.githubusercontent.com/ShindeOp/dp_machine/master/data.csv"

try:
    # Load CSV with correct separator
    df = pd.read_csv(csv_url, sep=";")
    st.success("Dataset loaded successfully!")

    # Show first 10 rows
    with st.expander("📂 Preview Data"):
        st.dataframe(df.head(10))

    # Define target (y) and features (X)
    y = df["Target"]
    X = df.drop("Target", axis=1)

    # Encode categorical features and target if needed
    X = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)
    y = LabelEncoder().fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained! Accuracy: {acc:.2f}")

    # Prediction demo with user input
    with st.expander("🎯 Try Prediction"):
        sample = {}
        for col in X.columns:
            val = st.number_input(f"Enter {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
            sample[col] = val

        if st.button("Predict"):
            sample_df = pd.DataFrame([sample])
            pred = model.predict(sample_df)[0]
            st.info(f"Prediction: {'Dropout' if pred == 0 else 'Graduate'}")

except Exception as e:
    st.error(f"Error loading dataset: {e}")
