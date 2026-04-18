import streamlit as st
import pandas as pd

# Title
st.title("📊 Data Sense 2.0 - Bakery Dataset")

# File uploader (optional but useful)
uploaded_file = st.file_uploader("Worked dataset- DataSense", type=["csv", "xlsx"])

if uploaded_file is not None:
    
    # Read file depending on type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Show data
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Basic info
    st.subheader("Dataset Info")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # Columns
    st.subheader("Columns")
    st.write(df.columns.tolist())

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

else:
    st.info("Please upload the 'Worked dataset- DataSense' file")
