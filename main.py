import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# SETUP
# ----------------------------
st.title("🚲 Bike Buyers KPI Dashboard (YES Only)")

df = pd.read_excel(
    "Worked dataset- DataSense.xlsx",
    sheet_name="Working sheet"
)

df.columns = df.columns.str.strip()

target = "Purchased Bike"

# Filter only buyers
buyers = df[df[target] == "Yes"]

# ----------------------------
# KPI CARDS
# ----------------------------
st.subheader("📊 Overview (Buyers Only)")

col1, col2 = st.columns(2)

col1.metric("Total Buyers", len(buyers))
col2.metric("Share of Dataset", f"{len(buyers) / len(df) * 100:.2f}%")

st.divider()

# ----------------------------
# FUNCTION: YES ONLY CHART
# ----------------------------
def plot_yes_only(feature):
    st.subheader(f"Bike Buyers (YES) - {feature}")

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

# ----------------------------
# INCOME (special case)
# ----------------------------
if "Income" in df.columns:
    st.subheader("💰 Income Distribution (Buyers Only)")

    fig, ax = plt.subplots()
    buyers["Income"].plot(kind="hist", bins=10, ax=ax)

    ax.set_xlabel("Income")

    st.pyplot(fig)
