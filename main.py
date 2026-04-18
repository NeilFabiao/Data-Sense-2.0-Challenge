import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("🚲 Bike Buyers Analysis - Side by Side Pie Charts")

# Load dataset
df = pd.read_excel(
    "Worked dataset- DataSense.xlsx",
    sheet_name="Working sheet"
)

df.columns = df.columns.str.strip()

target = "Purchased Bike"

# Filter only buyers
buyers = df[df[target] == "Yes"]

# ----------------------------
# FUNCTION: PIE CHART
# ----------------------------
def create_pie(ax, feature):
    data = buyers[feature].value_counts()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(feature)
    ax.axis("equal")

# ----------------------------
# SIDE BY SIDE LAYOUT
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    create_pie(ax1, "Gender")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    create_pie(ax2, "Marital Status")
    st.pyplot(fig2)

# ----------------------------
# ANOTHER ROW (optional)
# ----------------------------
col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots()
    create_pie(ax3, "Region")
    st.pyplot(fig3)

with col4:
    fig4, ax4 = plt.subplots()
    create_pie(ax4, "Education")
    st.pyplot(fig4)
