import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("🚲 Bike Buyers Analysis (YES Only)")

# Load dataset
df = pd.read_excel(
    "Worked dataset- DataSense.xlsx",
    sheet_name="Working sheet"
)

df.columns = df.columns.str.strip()

target = "Purchased Bike"

# Filter ONLY buyers
buyers = df[df[target] == "Yes"]

# ----------------------------
# FUNCTION: YES ONLY PLOT
# ----------------------------
def plot_yes_only(feature):
    st.subheader(f"Bike Buyers (YES) by {feature}")

    data = buyers[feature].value_counts()

    fig, ax = plt.subplots()
    data.plot(kind="bar", ax=ax)

    ax.set_ylabel("Number of Buyers")

    st.pyplot(fig)

# ----------------------------
# KPIs (YES ONLY)
# ----------------------------
if "Gender" in df.columns:
    plot_yes_only("Gender")

if "Marital Status" in df.columns:
    plot_yes_only("Marital Status")

if "Education" in df.columns:
    plot_yes_only("Education")

if "Occupation" in df.columns:
    plot_yes_only("Occupation")

if "Region" in df.columns:
    plot_yes_only("Region")

if "Commute Distance" in df.columns:
    plot_yes_only("Commute Distance")

if "Age brackets" in df.columns:
    plot_yes_only("Age brackets")

if "Income" in df.columns:
    st.subheader("Income of Bike Buyers (Distribution)")
    fig, ax = plt.subplots()
    buyers["Income"].plot(kind="hist", bins=10, ax=ax)
    st.pyplot(fig)if "Gender" in df.columns:
    plot_yes_no("Gender")

# ----------------------------
# KPI 2: Marital Status
# ----------------------------
if "Marital Status" in df.columns:
    plot_yes_no("Marital Status")

# ----------------------------
# KPI 3: Education
# ----------------------------
if "Education" in df.columns:
    plot_yes_no("Education")

# ----------------------------
# KPI 4: Occupation
# ----------------------------
if "Occupation" in df.columns:
    plot_yes_no("Occupation")

# ----------------------------
# KPI 5: Region
# ----------------------------
if "Region" in df.columns:
    plot_yes_no("Region")

# ----------------------------
# KPI 6: Commute Distance
# ----------------------------
if "Commute Distance" in df.columns:
    plot_yes_no("Commute Distance")

# ----------------------------
# KPI 7: Age Brackets
# ----------------------------
if "Age brackets" in df.columns:
    plot_yes_no("Age brackets")
