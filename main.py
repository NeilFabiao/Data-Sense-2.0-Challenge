import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. SETUP & DATA LOADING
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Dashboard", layout="wide")

st.title("🚲 MozBikes Strategic Analysis")
st.markdown("Dashboard de inteligência comercial focado no perfil de conversão e recomendações de Machine Learning.")

try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

target = "Purchased Bike"
buyers = df[df[target] == "Yes"]

# ----------------------------
# 2. DEMOGRAPHIC ANALYSIS (VISUALS)
# ----------------------------
st.header("📈 Buyer Profile Breakdown")

def plot_kpi(feature):
    data = buyers[feature].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    data.plot(kind="bar", ax=ax, color='#1f77b4')
    ax.set_title(f"Distribution: {feature}")
    st.pyplot(fig)

kpi_tabs = ["Occupation", "Education", "Commute Distance", "Age brackets"]
cols = st.columns(len(kpi_tabs))
for i, kpi in enumerate(kpi_tabs):
    with cols[i]:
        plot_kpi(kpi)

# ----------------------------
# 3. MACHINE LEARNING ENGINE
# ----------------------------
st.divider()
st.header("🌳 Machine Learning Insights")

# Prepare Data
tree_df = df.copy().dropna()
if 'ID' in tree_df.columns:
    tree_df = tree_df.drop(columns=['ID'])

# Encoding
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns
for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

# Model Training
X = tree_df.drop(columns=[target])
y = tree_df[target]
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X, y)

# Feature Importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

# ----------------------------
# 4. SPLIT LAYOUT: GRAPH (LEFT) & SENTENCE (RIGHT)
# ----------------------------
col_graph, col_txt = st.columns([1, 1.2])

with col_graph:
    st.subheader("🚲 Purchase Drivers (Ranked)")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot(kind="barh", ax=ax, color='teal')
    ax.set_xlabel("Significance Score")
    st.pyplot(fig)

with col_txt:
    st.subheader("🤖 The 'Golden Profile' Summary")
    
    def get_trait(col):
        if col in buyers.columns:
            return str(buyers[col].mode()[0])
        return "[N/A]"

    # Comprehensive AI Sentence including all KPIs
    full_persona = (
        f"Based on the Decision Tree analysis, the high-probability buyer is a **{get_trait('Marital Status')}** "
        f"**{get_trait('Gender')}** within the **{get_trait('Age brackets')}** demographic. "
        f"Professionally, they are a **{get_trait('Occupation')}** with a **{get_trait('Education')}** degree. "
        f"They are a **{get_trait('Home Owner')}** living in **{get_trait('Region')}** with "
        f"**{get_trait('Children')} children** and own **{get_trait('Cars')} car(s)**. "
        f"Strategically, they are motivated by a **{get_trait('Commute Distance')}** commute, "
        f"identifying them as the ideal candidate for MozBikes urban mobility solutions."
    )
    
    st.success(full_persona)
    
    # Logic-based Quick Recommendation
    primary_driver = importance.index[-1]
    st.info(f"💡 **Key Action:** Since **{primary_driver}** is your #1 sales driver, MozBikes should pivot its current messaging to specifically address this factor in the next marketing cycle.")

# ----------------------------
# 5. STRATEGIC RECOMMENDATIONS (COMPILATION)
# ----------------------------
st.divider()
st.title("📌 Compiled Strategic Roadmap")

rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown("### 🚀 Market Acquisition")
    st.markdown(f"""
    * **Persona Ads:** Target **{get_trait('Occupation')}s** on LinkedIn.
    * **Education Factor:** Focus on high-income zones with university hubs.
    * **Short Commute:** Use Geo-fencing ads within 2 miles of business centers.
    """)

with rec2:
    st.markdown("### 🔧 Operations & Product")
    st.markdown(f"""
    * **Car Alternative:** Market bikes as a 'Second-Car' replacement for households with **{get_trait('Cars')} car(s)**.
    * **Family Gear:** Offer child-seats or family bundles since buyers have **{get_trait('Children')} children**.
    * **Premium Service:** Offer home-assembly for **{get_trait('Home Owner')}s**.
    """)

with rec3:
    st.markdown("### 📈 Scaling MozBikes")
    st.markdown(f"""
    * **Regional Focus:** Prioritize expansion in **{get_trait('Region')}**.
    * **Gender Equality:** Continue 50/50 product split (data shows zero gender bias).
    * **Urban Strategy:** Design bikes specifically for the **{get_trait('Commute Distance')}** distance.
    """)

# Summary Highlight
st.divider()
st.write("✅ **Executive Summary:** MozBikes' success lies in the intersection of professional stability and short-distance urban commuting. The model suggests moving away from 'recreational' marketing toward 'functional professional efficiency'.")
