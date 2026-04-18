import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# SETUP
# ----------------------------
st.title("🚲 Bike Buyers Analysis (YES Only) - Bar & Pie Charts")

df = pd.read_excel(
    "Worked dataset- DataSense.xlsx",
    sheet_name="Working sheet"
)

df.columns = df.columns.str.strip()

target = "Purchased Bike"

# Filter only buyers
buyers = df[df[target] == "Yes"]

# ----------------------------
# FUNCTION: BAR + PIE SIDE BY SIDE
# ----------------------------
def plot_bar_pie(feature):
    st.subheader(f"📊 {feature} - Bike Buyers (YES)")

    data = buyers[feature].value_counts()

    col1, col2 = st.columns(2)

    # ---------------- BAR CHART ----------------
    with col1:
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Bar Chart")
        st.pyplot(fig)

    # ---------------- PIE CHART ----------------
    with col2:
        fig, ax = plt.subplots()
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Pie Chart")
        ax.axis("equal")
        st.pyplot(fig)

# ----------------------------
# KPIs
# ----------------------------
if "Gender" in df.columns:
    plot_bar_pie("Gender")

if "Marital Status" in df.columns:
    plot_bar_pie("Marital Status")

if "Education" in df.columns:
    plot_bar_pie("Education")

if "Occupation" in df.columns:
    plot_bar_pie("Occupation")

if "Region" in df.columns:
    plot_bar_pie("Region")

if "Commute Distance" in df.columns:
    plot_bar_pie("Commute Distance")

if "Age brackets" in df.columns:
    plot_bar_pie("Age brackets")
