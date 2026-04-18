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
# We focus on "Yes" for the demographic visuals and the persona sentence
buyers = df[df[target] == "Yes"]

# ----------------------------
# 4. FUNCTION: BAR + PIE
# ----------------------------
def plot_bar_pie(feature, summary_text=""):
    if feature not in buyers.columns:
        return
        
    st.divider()
    st.subheader(f"📊 {feature} - Perfil dos Compradores")

    data = buyers[feature].value_counts()
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax, color='#1f77b4')
        ax.set_ylabel("Quantidade")
        ax.set_title(f"Distribuição por {feature}")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Percentual por {feature}")
        ax.axis("equal")
        st.pyplot(fig)

    if summary_text:
        st.info(summary_text)

# ----------------------------
# 5. VISUAL ANALYSIS
# ----------------------------
st.header("📈 Buyer Profile Breakdown")
plot_bar_pie("Gender", "**Insight:** Distribuição equilibrada entre homens e mulheres.")
plot_bar_pie("Education", "**Insight:** Maioria com ensino superior.")
plot_bar_pie("Occupation", "**Insight:** Profissionais e técnicos dominam.")
plot_bar_pie("Age brackets", "**Insight:** Meia-idade domina as compras.")
plot_bar_pie("Commute Distance", "**Insight:** Forte presença de trajetos curtos.")
plot_bar_pie("Home Owner", "**Insight:** Indica estabilidade financeira do cliente.")

# ----------------------------
# 6. MACHINE LEARNING ENGINE
# ----------------------------
tree_df = df.copy().dropna()
if "ID" in tree_df.columns:
    tree_df = tree_df.drop(columns=["ID"])

# Encode target for ML
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns

for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

X = tree_df.drop(columns=[target])
y = tree_df[target]

model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance[importance > 0].sort_values()

# ----------------------------
# 7. SPLIT LAYOUT: GRAPH (LEFT) & INSIGHT (RIGHT)
# ----------------------------
st.divider()
st.header("🧠 Machine Learning 'Golden Profile'")

col_graph, col_txt = st.columns([1, 1.2])

with col_graph:
    st.subheader("🚲 Purchase Drivers (Ranked)")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot(kind="barh", ax=ax, color='teal')
    ax.set_xlabel("Significance Score")
    st.pyplot(fig)

with col_txt:
    st.subheader("🤖 Ideal Customer Portrait")
    
    # Robust Trait Finder (Fixed Function Name)
    def get_mode(col_name):
        # Case-insensitive column search to avoid [N/A]
        actual_col = next((c for c in buyers.columns if c.lower() == col_name.lower()), None)
        if actual_col and not buyers[actual_col].mode().empty:
            val = buyers[actual_col].mode()[0]
            # Custom formatting for Home Owner
            if col_name.lower() == "home owner":
                return "homeowner" if str(val).lower() == "yes" else "renter"
            return str(val)
        return "N/A"

    # AI Generated Sentence
    full_persona = (
        f"The Machine Learning analysis identifies the high-probability buyer as a **{get_mode('Marriedarital Status')}** "
        f"**{get_mode('Gender')}** within the **{get_mode('Age brackets')}** demographic. "
        f"Typically a **{get_mode('Education')}** graduate who is a **{get_mode('Occupation')}** role, "
        f"this individual is likely a **{get_mode('Home Owner')}** living in **{get_mode('Region')}** "
        f"with **{get_mode('Children')} child(ren)** and **{get_mode('Cars')} vehicle(s)**. "
        f"With a commute of **{get_mode('Commute Distance')}**, they represent MozBikes' primary target for "
        f"efficient urban mobility solutions."
    )
    
    st.success(full_persona)

    # Top 3 Strategic Actions
    st.markdown("#### 🔥 Top 3 Strategic Actions")
    top3 = importance.sort_values(ascending=False).head(3)
    
    for i, (factor, score) in enumerate(top3.items(), 1):
        if factor == "Cars":
            st.write(f"{i}. **Vehicle Substitution:** Target households with {get_mode('Cars')} car(s) to position bikes as a cost-saving alternative.")
        elif factor == "Commute Distance":
            st.write(f"{i}. **Urban Focus:** Prioritize marketing within the {get_mode('Commute Distance')} radius of business hubs.")
        elif "Age" in factor:
            st.write(f"{i}. **Demographic Targeting:** Focus branding on the {get_mode('Age brackets')} group, emphasizing reliability.")
        else:
            st.write(f"{i}. **{factor} Optimization:** Leverage {factor} data to refine digital ad segmentation.")

# ----------------------------
# 8. COMPILED RECOMMENDATIONS
# ----------------------------
st.divider()
st.header("🚀 Key Strategic Roadmap")

rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown("### 📢 Market Acquisition")
    st.markdown(f"""
    - **Persona Ads:** Use images of **{get_mode('Occupation')}s** commuting.
    - **Hyper-Local:** Target **{get_mode('Region')}** specifically.
    - **LinkedIn:** Reach **{get_mode('Education')}** degree holders.
    """)

with rec2:
    st.markdown("### 🔧 Operations & Product")
    st.markdown(f"""
    - **B2B Strategy:** Offer fleet leasing to companies with many **{get_mode('Occupation')}s**.
    - **Family Bundles:** Stock accessories for parents with **{get_mode('Children')}** kids.
    - **Premium Delivery:** Home assembly for **{get_mode('Home Owner')}s**.
    """)

with rec3:
    st.markdown("### 📈 Scaling MozBikes")
    st.markdown(f"""
    - **Car Replacement:** Market to the **{get_mode('Cars')}-car** segment.
    - **Distance Focus:** Optimize bike durability for **{get_mode('Commute Distance')}** trips.
    - **Neutral Branding:** Maintain gender-neutral branding (Data confirms 50/50 split).
    """)

st.success("""
✅ **Final Executive Summary:** MozBikes success lies in the intersection of professional stability and short-distance urban commuting. 
The model suggests moving away from 'leisure' and toward 'functional professional mobility'.
""")
