import streamlit as st
import pandas as pd

st.title('🎈 ML app')
st.info('This is an ML app')

# Raw URL to the CSV
csv_url = "https://raw.githubusercontent.com/ShindeOp/dp_machine/master/data.csv"

try:
    # Attempt to load the CSV file
    df = pd.read_csv(csv_url)
    st.success("Dataset loaded successfully!")
    st.dataframe(df)  # Display dataframe
except Exception as e:
    # If an error occurs, display it in Streamlit
    st.error(f"Error loading dataset: {e}")
