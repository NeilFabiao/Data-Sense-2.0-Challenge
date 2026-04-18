import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# SETUP
# ----------------------------
st.title("🚲 Bike Purchase KPI Dashboard")

df = pd.read_excel(
    "Worked dataset- DataSense.xlsx",
    sheet_name="Working sheet"
)

df.columns = df.columns.str.strip()

target = "Purchased Bike"

# ----------------------------
# KPI CARDS (OVERVIEW)
# ----------------------------
total_customers = len(df)
buyers = df[df[target] == "Yes"]
non_buyers = df[df[target] == "No"]

conversion_rate = len(buyers) / total_customers * 100

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Total Buyers", len(buyers))
col3.metric("Conversion Rate", f"{conversion_rate:.2f}%")

st.divider()

# ----------------------------
# FUNCTION: YES/NO SPLIT CHART
# ----------------------------
def plot_kpi(feature):
    st.subheader(f"{feature} vs Bike Purchase")

    ct = pd.crosstab(df[feature], df[target])

    fig, ax = plt.subplots()
    ct.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel("Count")
    ax.legend(title="Purchased Bike")

    st.pyplot(fig)

# ----------------------------
# KPI SECTIONS
# ----------------------------

if "Gender" in df.columns:
    plot_kpi("Gender")

if "Marital Status" in df.columns:
    plot_kpi("Marital Status")

if "Education" in df.columns:
    plot_kpi("Education")

if "Occupation" in df.columns:
    plot_kpi("Occupation")

if "Region" in df.columns:
    plot_kpi("Region")

if "Commute Distance" in df.columns:
    plot_kpi("Commute Distance")

if "Age brackets" in df.columns:
    plot_kpi("Age brackets")

if "Income" in df.columns:
    st.subheader("Income Distribution vs Purchase")

    fig, ax = plt.subplots()
    df.boxplot(column="Income", by=target, ax=ax)

    st.pyplot(fig)
