import streamlit as st
import pandas as pd

# Title
st.title("📊 Data Sense 2.0 - Bakery Dataset")

# Load Excel file
df = pd.read_excel("Worked dataset- DataSense.xlsx")


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
