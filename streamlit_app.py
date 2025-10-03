import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# ------------------------------------------------------
# 1. Streamlit Page Config
# ------------------------------------------------------
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")
st.title("🎓 Student Dropout Prediction App")
st.info("This app predicts whether a student will Dropout, Graduate, or remain Enrolled.")

# ------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------
csv_url = "https://raw.githubusercontent.com/ASIF-Kh/Student-Dropout-Prediction/main/data.csv"
prediction_output_container = st.empty()

try:
    df = pd.read_csv(csv_url, sep=";")
    st.success("✅ Dataset loaded successfully!")

    with st.expander("📂 Preview Data"):
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        rows_to_show = st.slider("Rows to preview:", 5, len(df), 10)
        st.dataframe(df.head(rows_to_show))
        st.write("Column Info:")
        st.write(df.dtypes)

    # ------------------------------------------------------
    # 3. Preprocessing
    # ------------------------------------------------------
    y_original = df["Target"]
    X_original = df.drop("Target", axis=1)

    feature_encoders = {}
    X_encoded = X_original.copy()

    # Identify categorical features
    categorical_cols = X_original.select_dtypes(include="object").columns

    # ✅ FIXED BLOCK (brackets properly closed)
    discrete_numeric_cols = [
        col for col in X_original.columns
        if X_original[col].dtype != "object"
        and not np.issubdtype(X_original[col].dtype, np.floating)
        and X_original[col].nunique() < 50
    ]

    # Combine categorical + discrete numeric
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

    # ------------------------------------------------------
    # 4. Train/Test Split
    # ------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------
    # 5. Model Training (Random Forest)
    # ------------------------------------------------------
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Random Forest trained successfully! Test Accuracy: **{acc:.2f}**")

    # ------------------------------------------------------
    # 6. Prediction UI
    # ------------------------------------------------------
    with st.expander("🎯 Try Prediction (Input Features)", expanded=True):
        st.write("Adjust the features below to get a prediction for a potential student.")

        sample_encoded = {}
        cols = st.columns(2)
        col_index = 0

        for col in X_original.columns:
            with cols[col_index % 2]:
                if col in feature_encoders:
                    le = feature_encoders[col]
                    options = le.classes_
                    default_index = list(options).index(str(X_original[col].mode().iloc[0]))

                    selected = st.selectbox(
                        f"{col}", options=options, index=default_index, key=f"sb_{col}"
                    )
                    sample_encoded[col] = le.transform([selected])[0]
                else:
                    data_col = X_original[col]
                    min_val = float(data_col.min())
                    max_val = float(data_col.max())
                    is_float = np.issubdtype(data_col.dtype, np.floating)

                    if is_float:
                        mean_val = float(data_col.mean())
                        step = 0.1
                        fmt = "%.2f"
                    else:
                        mean_val = float(int(data_col.mean()))
                        step = 1.0
                        fmt = "%d"

                    val = st.number_input(
                        f"{col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=step,
                        format=fmt,
                        key=f"ni_{col}",
                    )
                    sample_encoded[col] = val
            col_index += 1

        st.markdown("---")
        button_clicked = st.button("PREDICT STUDENT OUTCOME", type="primary", use_container_width=True)

    # ------------------------------------------------------
    # 7. Prediction Output
    # ------------------------------------------------------
    if button_clicked:
        sample_df = pd.DataFrame([sample_encoded])
        sample_df = sample_df[X_encoded.columns]  # Match column order

        pred_encoded = model.predict(sample_df)[0]
        predicted_label = target_labels.get(pred_encoded, "Unknown Outcome")
        is_dropout = "Dropout" in predicted_label

        with prediction_output_container.container():
            st.subheader("Prediction Results:")
            st.markdown("---")

            st.subheader("Dropout Status:")
            if is_dropout:
                st.error("❌ **PREDICTED DROPOUT**")
            else:
                st.success("✅ **NOT PREDICTED DROPOUT**")

            st.subheader("Detailed Prediction:")
            if is_dropout:
                st.error(f"⚠️ **Predicted Outcome: {predicted_label}**")
                st.caption("This student is highly likely to drop out. Consider intervention.")
            elif "Graduate" in predicted_label:
                st.balloons()
                st.success(f"🎉 **Predicted Outcome: {predicted_label}**")
                st.caption("This student is likely to graduate successfully.")
            else:
                st.info(f"➡️ **Predicted Outcome: {predicted_label}**")
                st.caption("This student is predicted to remain enrolled.")

except Exception as e:
    prediction_output_container.empty()
    st.error("❌ An error occurred while loading data or training the model.")
    st.caption("Please ensure the CSV URL is correct and accessible.")
    st.exception(e)
