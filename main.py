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



from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.divider()
st.title("🌳 What Drives Bike Purchases (Decision Tree)")

tree_df = df.copy()
tree_df = tree_df.dropna()

# ----------------------------
# Encode ONLY categorical columns
# (DO NOT touch numeric ones like Income, Cars, Age)
# ----------------------------
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})

categorical_cols = tree_df.select_dtypes(include="object").columns

for col in categorical_cols:
    tree_df[col] = LabelEncoder().fit_transform(tree_df[col])

# ----------------------------
# FEATURES & TARGET
# ----------------------------
X = tree_df.drop(columns=[target])
y = tree_df[target]

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# ----------------------------
# FEATURE IMPORTANCE (YES DRIVERS)
# ----------------------------
st.subheader("🚲 Key Drivers of Bike Purchase")

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values()

fig, ax = plt.subplots()
importance.plot(kind="barh", ax=ax)
ax.set_title("Factors influencing Bike Purchase (YES)")
st.pyplot(fig)

# ----------------------------
# TOP 5 FACTORS
# ----------------------------
st.subheader("🔥 Top 5 Drivers of Purchase")

st.write(importance.sort_values(ascending=False).head(5))
