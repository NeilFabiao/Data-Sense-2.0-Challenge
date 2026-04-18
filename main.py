import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. SETUP
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Dashboard", layout="wide")

st.title("🚲 MozBikes Strategic Analysis")
st.markdown("Dashboard de inteligência comercial focado no perfil de conversão e recomendações de Machine Learning.")

# ----------------------------
# 2. LOAD DATA
# ----------------------------
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

target = "Purchased Bike"

# ----------------------------
# 3. FILTER BUYERS
# ----------------------------
buyers = df[df[target] == "Yes"]

# ----------------------------
# 4. VISUAL ANALYSIS
# ----------------------------
st.header("📈 Buyer Profile Breakdown")

def plot_kpi(feature):
    if feature in buyers.columns:
        data = buyers[feature].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        data.plot(kind="bar", ax=ax, color='#1f77b4')
        ax.set_title(f"Distribution: {feature}")
        st.pyplot(fig)

cols = st.columns(3)

features = ["Gender", "Education", "Occupation", "Age brackets", "Commute Distance"]

for i, feature in enumerate(features):
    with cols[i % 3]:
        plot_kpi(feature)

# ----------------------------
# 5. MACHINE LEARNING
# ----------------------------
st.divider()
st.header("🌳 What Drives Bike Purchases")

tree_df = df.copy().dropna()

# Drop irrelevant columns
if "ID" in tree_df.columns:
    tree_df = tree_df.drop(columns=["ID"])

# Encode target
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})

# Encode categoricals
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns

for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

# Features / Target
X = tree_df.drop(columns=[target])
y = tree_df[target]

# Train model
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X, y)

# ----------------------------
# 6. FEATURE IMPORTANCE
# ----------------------------
st.subheader("🚲 Key Drivers of Purchase")

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance[importance > 0].sort_values()

fig, ax = plt.subplots()
importance.plot(kind="barh", ax=ax, color='teal')
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# ----------------------------
# 7. TOP 3 DRIVERS
# ----------------------------
st.subheader("🔥 Top Drivers")

top3 = importance.sort_values(ascending=False).head(3)

for i, (factor, score) in enumerate(top3.items(), 1):
    st.write(f"{i}️⃣ **{factor}** — Impact: {score:.2%}")

# ----------------------------
# 8. PERSONA (FIXED - NO N/A)
# ----------------------------
st.divider()
st.header("🧠 Ideal Customer Profile")

# Clean original dataset for persona
persona_df = df.copy().dropna()

if "ID" in persona_df.columns:
    persona_df = persona_df.drop(columns=["ID"])

persona_df = persona_df[persona_df[target] == "Yes"]

def get_mode(col):
    if col in persona_df.columns and not persona_df[col].mode().empty:
        return persona_df[col].mode()[0]
    return "N/A"

persona = {
    "Gender": get_mode("Gender"),
    "Age": get_mode("Age brackets"),
    "Occupation": get_mode("Occupation"),
    "Education": get_mode("Education"),
    "Region": get_mode("Region"),
    "Children": get_mode("Children"),
    "Cars": get_mode("Cars"),
    "Commute": get_mode("Commute Distance")
}

st.markdown(f"""
Based on the Decision Tree analysis and buyer distribution, the **highest-probability MozBikes customer** is a **{persona['Age']} {persona['Gender']} professional**.

This individual typically:
- Works in a **{persona['Occupation']} role**
- Holds a **{persona['Education']} qualification**
- Lives in **{persona['Region']}**
- Has **{persona['Children']} children** and owns **{persona['Cars']} car(s)**

From a behavioral standpoint, their **{persona['Commute']} commute** strongly indicates a preference for **short-distance, efficient transportation**.

👉 This profile represents the **core urban commuter segment**, making them the ideal target for MozBikes' mobility solutions.
""")

# ----------------------------
# 9. STRATEGIC ACTIONS
# ----------------------------
st.divider()
st.header("🚀 Key Strategic Actions")

st.markdown("""
Based on the model and buyer analysis, MozBikes should prioritize the following **top 3 drivers of purchase behavior**:

### 1️⃣ Vehicle Ownership (Cars)
- Customers with fewer cars are more likely to purchase bikes  
- **Action:** Position bikes as a *practical alternative to cars*  
- Messaging: “Save money, avoid traffic, simplify your commute”

### 2️⃣ Commute Distance
- Short-distance commuters dominate buyers  
- **Action:** Focus on **urban mobility & last-mile transport**  
- Messaging: “Perfect for quick daily trips”

### 3️⃣ Occupation (Working Professionals)
- Professionals and skilled workers lead purchases  
- **Action:** Target workplaces:
  - Corporate partnerships  
  - Employee mobility programs  

---

💡 **Strategic Insight:**  
MozBikes is not just selling bikes — it is solving a **daily transport problem**.

Success depends on aligning with:
- Urban lifestyles  
- Short commutes  
- Cost-conscious consumers  
""")

# ----------------------------
# 10. FINAL TAKEAWAY
# ----------------------------
st.success("""
✅ **Final Takeaway:**  
The ideal MozBikes customer is a working professional with a short commute, using bicycles as a practical alternative to cars.  
Focusing on this segment will maximize conversion and growth.
""")
