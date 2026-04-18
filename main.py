import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("🚲 Bike Purchase Analysis - Yes/No Breakdown")

# Load dataset
df = pd.read_excel(
    "Worked dataset- DataSense.xlsx",
    sheet_name="Working sheet"
)

df.columns = df.columns.str.strip()

target = "Purchased Bike"

# ----------------------------
# FUNCTION: YES/NO SPLIT PLOT
# ----------------------------
def plot_yes_no(feature):
    st.subheader(f"{feature} vs Purchased Bike (Yes/No)")

    ct = pd.crosstab(df[feature], df[target])

    fig, ax = plt.subplots()
    ct.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel("Count")
    ax.legend(title="Purchased Bike")

    st.pyplot(fig)

# ----------------------------
# KPI 1: Gender
# ----------------------------
if "Gender" in df.columns:
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
