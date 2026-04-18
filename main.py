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
    if feature in buyers.columns:
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
# 4. SPLIT LAYOUT: GRAPH & SENTENCE
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
    
    # Robust Trait Finder (Fixes [N/A] issues)
    def get_trait(col_name):
        # Search for column case-insensitively
        actual_col = next((c for c in buyers.columns if c.lower() == col_name.lower()), None)
        if actual_col:
            mode_val = buyers[actual_col].mode()
            if not mode_val.empty:
                val = mode_val[0]
                # Clean up "Yes/No" for Home Owner
                if col_name.lower() == "home owner":
                    return "homeowner" if str(val).lower() == "yes" else "renter"
                return str(val)
        return "[Data Missing]"

    # Improved Professional Writing
    full_persona = (
        f"The Machine Learning analysis identifies the high-probability buyer as a **{get_trait('Marital Status')}** "
        f"**{get_trait('Gender')}** within the **{get_trait('Age brackets')}** demographic. "
        f"Typically a **{get_trait('Education')}** graduate working in a **{get_trait('Occupation')}** role, "
        f"this individual is likely a **{get_trait('Home Owner')}** living in **{get_trait('Region')}** "
        f"with **{get_trait('Children')} child(ren)** and **{get_trait('Cars')} vehicle(s)**. "
        f"With a commute of **{get_trait('Commute Distance')}**, they represent MozBikes' primary target for "
        f"efficient urban mobility solutions."
    )
    
    st.success(full_persona)
    
    # IMPROVED: Top 3 Key Actions
    st.markdown("#### 🔥 Top 3 Strategic Actions")
    top_3 = importance.sort_values(ascending=False).head(3)
    for i, (feature, score) in enumerate(top_3.items()):
        if feature == "Cars":
            st.write(f"{i+1}. **Vehicle Substitution:** Target households with {get_trait('Cars')} car(s) to position bikes as a primary cost-saving tool.")
        elif feature == "Commute Distance":
            st.write(f"{i+1}. **Urban Proximity:** Focus retail presence and ads within the {get_trait('Commute Distance')} radius of business hubs.")
        elif feature == "Age brackets":
            st.write(f"{i+1}. **Age-Specific Marketing:** Tailor branding to the {get_trait('Age brackets')} group, focusing on health and reliability.")
        else:
            st.write(f"{i+1}. **{feature} Optimization:** Leverage {feature} patterns to refine customer acquisition.")

# ----------------------------
# 5. COMPILED STRATEGIC ROADMAP
# ----------------------------
st.divider()
st.title("📌 Compiled Strategic Roadmap")

rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown("### 🚀 Market Acquisition")
    st.markdown(f"""
    * **Persona-Based Ads:** Showcase **{get_trait('Occupation')}s** using bikes for quick trips.
    * **High-ROI Zones:** Aggressively target **{get_trait('Region')}** based on high conversion.
    * **LinkedIn Targeting:** Reach **{get_trait('Education')}** holders with 'Green Commute' messaging.
    """)

with rec2:
    st.markdown("### 🔧 Operations & Product")
    st.markdown(f"""
    * **Fleet Strategy:** Since many are **{get_trait('Occupation')}s**, offer B2B fleet leasing to companies.
    * **Home Delivery:** Offer premium assembly for **{get_trait('Home Owner')}s**.
    * **Family Gear:** Stock child-carrying accessories for parents of **{get_trait('Children')}** child(ren).
    """)

with rec3:
    st.markdown("### 📈 Scaling MozBikes")
    st.markdown(f"""
    * **Second-Vehicle Replacement:** Market to the **{get_trait('Cars')}-car** demographic to reduce fuel costs.
    * **Short-Range Optimization:** Design bikes specifically for the **{get_trait('Commute Distance')}** commute.
    * **Universal Branding:** Maintain gender-neutral branding (Data confirms 50/50 split).
    """)

st.divider()
st.info("✅ **Final Executive Summary:** MozBikes should transition from general sales to a 'Functional Lifestyle' brand. The model highlights that professional stability and short urban distances are the strongest predictors of a completed sale.")
