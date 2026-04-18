import streamlit as st
import pandas as pd

# Title
st.title("📊 Data Sense 2.0 - Bakery Dataset")

# Load dataset directly (make sure file is in same folder)
df = pd.read_csv("Worked dataset- DataSense.csv")

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
