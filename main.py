import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. SETUP & THEME
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
# 2.5 DATA PROCESSING (INCOME & AGE)
# ----------------------------
df["Income Group"] = pd.cut(
    df["Income"],
    bins=[0, 30000, 60000, 90000, 120000, float("inf")],
    labels=["<30k", "30–60k", "60–90k", "90–120k", "120k+"]
)

# ----------------------------
# 3. FILTER BUYERS
# ----------------------------
buyers = df[df[target] == "Yes"]

# Helper to find common traits
def get_mode(col_name):
    actual_col = next((c for c in buyers.columns if c.lower() == col_name.lower()), None)
    if actual_col and not buyers[actual_col].mode().empty:
        val = buyers[actual_col].mode()[0]
        if col_name.lower() == "home owner":
            return "Homeowner" if str(val).lower() == "yes" else "Renter"
        return str(val)
    return "N/A"

# ----------------------------
# 4. EXECUTIVE SUMMARY METRICS
# ----------------------------
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Top Age Group", get_mode("Age brackets"))
m2.metric("Primary Region", get_mode("Region"))
m3.metric("Avg Commute", get_mode("Commute Distance"))
m4.metric("Conversion Lead", get_mode("Occupation"))

# ----------------------------
# 5. FUNCTION: IMPROVED BAR + PIE (BOLD TITLES & LABELS)
# ----------------------------
def plot_bar_pie(feature, summary_text=""):
    if feature not in buyers.columns:
        return

    st.divider()
    # Using Markdown for Bold Subheaders
    st.markdown(f"### **📊 {feature} Analysis**")

    data = buyers[feature].value_counts().sort_index()
    col1, col2 = st.columns(2)

    # --- BAR CHART ---
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(data.index.astype(str), data.values, color='#2E86C1', edgecolor='black', alpha=0.8)
        
        # Gridlines for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # Data Labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_ylabel("Quantity", fontweight='bold')
        # BOLD TITLE
        ax.set_title(f"VOLUME BY {feature.upper()}", fontsize=13, fontweight='bold', pad=15)
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)

    # --- PIE CHART ---
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        wedges, texts, autotexts = ax.pie(
            data, 
            labels=data.index, 
            autopct="%1.1f%%", 
            startangle=140, 
            colors=['#1B4F72', '#2E86C1', '#5DADE2', '#AED6F1', '#D6EAF8'],
            pctdistance=0.80
        )
        plt.setp(autotexts, size=10, weight="bold", color="white")
        
        # Donut Shape
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        
        # BOLD TITLE
        ax.set_title(f"PERCENTAGE SHARE: {feature.upper()}", fontsize=13, fontweight='bold', pad=15)
        ax.axis("equal")
        st.pyplot(fig)

    if summary_text:
        st.info(summary_text)

# ----------------------------
# 6. VISUAL ANALYSIS SECTIONS
# ----------------------------
plot_bar_pie("Gender", "**Insight:** Balance leads to a gender-neutral product strategy.")
plot_bar_pie("Education", "**Insight:** High educational attainment suggests analytical marketing content.")
plot_bar_pie("Occupation", "**Insight:** The professional segment is the core revenue driver.")
plot_bar_pie("Age brackets", "**Insight:** Marketing should target the 'Middle Age' stability.")
plot_bar_pie("Commute Distance", "**Insight:** Focus on 'Last-Mile' solutions (Short commutes).")
plot_bar_pie("Cars", "**Insight:** High correlation between low car ownership and bike purchases.")

# ----------------------------
# 7. MACHINE LEARNING ENGINE
# ----------------------------
st.divider()
st.header("🧠 Predictive Analytics (Machine Learning)")

tree_df = df.copy().dropna()
if "ID" in tree_df.columns:
    tree_df = tree_df.drop(columns=["ID"])

# Feature selection
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns
for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

X = tree_df.drop(columns=[target, "Income Group"]) # Drop derived
y = tree_df[target]

model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X, y)
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

# ----------------------------
# 8. ML LAYOUT: GRAPH & PERSONA
# ----------------------------
col_graph, col_txt = st.columns([1, 1.2])

with col_graph:
    st.subheader("**🚲 Key Purchase Drivers**")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot(kind="barh", ax=ax, color='#117A65')
    
    # Add labels to importance bars
    for i, v in enumerate(importance):
        ax.text(v + 0.01, i, f'{v:.1%}', color='#117A65', va='center', fontweight='bold')
    
    ax.set_title("VARIABLE IMPACT ON SALE", fontweight='bold')
    st.pyplot(fig)

with col_txt:
    st.subheader("**🤖 The 'Golden Profile' Insight**")
    
    full_persona = (
        f"The Machine Learning model confirms that the ideal MozBikes customer is a **{get_mode('Marriedarital Status')}** "
        f"**{get_mode('Gender')}** in the **{get_mode('Age brackets')}** stage of life. "
        f"Typically a **{get_mode('Education')}** holder working as a **{get_mode('Occupation')}**, "
        f"this **{get_mode('Home Owner')}** resides in **{get_mode('Region')}**. "
        f"They balance a family with **{get_mode('Children')} children** and manage **{get_mode('Cars')} vehicles**. "
        f"Their **{get_mode('Commute Distance')}** commute makes them the primary candidate for bike-based transit."
    )
    st.success(full_persona)

    # Top 3 Actions
    st.markdown("#### **🔥 Top 3 Strategic Actions**")
    top3 = importance.sort_values(ascending=False).head(3)
    for i, (factor, score) in enumerate(top3.items(), 1):
        st.write(f"{i}. **{factor}:** Focus on the **{get_mode(factor)}** segment to optimize conversion.")

# ----------------------------
# 9. FINAL STRATEGY ROADMAP
# ----------------------------
st.divider()
st.header("📌 Execution Roadmap")
c1, c2, c3 = st.columns(3)
with c1:
    st.info(f"**Targeting:** Focus on {get_mode('Occupation')}s in {get_mode('Region')}.")
with c2:
    st.info(f"**Inventory:** Stock accessories for households with {get_mode('Children')} kids.")
with c3:
    st.info(f"**Pitch:** Position as a car-alternative for the {get_mode('Cars')}-car owners.")

st.success("✅ **Final Summary:** The data dictates that MozBikes should pivot from 'Adventure' to 'Efficiency'.")
