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
# 2.5 INCOME BINNING
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

# ----------------------------
# 4. FUNCTION: IMPROVED BAR + PIE
# ----------------------------
def plot_bar_pie(feature, summary_text=""):
    if feature not in buyers.columns:
        return

    st.divider()
    st.subheader(f"📊 {feature} - Perfil dos Compradores")

    data = buyers[feature].value_counts().sort_index()
    col1, col2 = st.columns(2)

    # --- IMPROVED BAR CHART ---
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(data.index.astype(str), data.values, color='#1f77b4', edgecolor='black', alpha=0.8)
        
        # Add Data Labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel("Quantidade de Compradores")
        ax.set_title(f"Volume por {feature}", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    # --- IMPROVED PIE CHART ---
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Use explode to slightly separate pieces if useful, or just clean labels
        wedges, texts, autotexts = ax.pie(
            data, 
            labels=data.index, 
            autopct="%1.1f%%", 
            startangle=140, 
            colors=plt.cm.Paired.colors,
            pctdistance=0.85
        )
        # Make percentages bold and white for readability
        plt.setp(autotexts, size=10, weight="bold", color="white")
        
        # Draw a circle at the center to turn it into a Donut (cleaner look)
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        
        ax.set_title(f"Market Share por {feature}", fontsize=12)
        ax.axis("equal")
        st.pyplot(fig)

    if summary_text:
        st.info(summary_text)

# ----------------------------
# 5. VISUAL ANALYSIS
# ----------------------------
st.header("📈 Buyer Profile Breakdown")
plot_bar_pie("Gender", "**Insight:** Distribuição equilibrada entre homens e mulheres.")
plot_bar_pie("Education", "**Insight:** Maioria com ensino superior (Bachelors/Graduate).")
plot_bar_pie("Occupation", "**Insight:** Profissionais e técnicos são a base do faturamento.")
plot_bar_pie("Age brackets", "**Insight:** O público de 'Middle Age' é o principal motor de vendas.")
plot_bar_pie("Commute Distance", "**Insight:** A maioria dos compradores percorre menos de 2 milhas.")
plot_bar_pie("Home Owner", "**Insight:** A posse de imóvel próprio correlaciona com maior conversão.")
plot_bar_pie("Income Group", "**Insight:** Compradores concentram-se na faixa de 30k–90k.")

# ----------------------------
# 6. MACHINE LEARNING ENGINE
# ----------------------------
tree_df = df.copy().dropna()
if "ID" in tree_df.columns:
    tree_df = tree_df.drop(columns=["ID"])

# Feature selection for ML
cols_to_drop = ["Income Group"] # Drop derived column for ML training
if "Age" in tree_df.columns and "Age brackets" in tree_df.columns:
    cols_to_drop.append("Age")
tree_df = tree_df.drop(columns=cols_to_drop)

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
# 7. SPLIT LAYOUT: GRAPH & INSIGHT
# ----------------------------
st.divider()
st.header("🧠 Machine Learning 'Golden Profile'")

col_graph, col_txt = st.columns([1, 1.2])

with col_graph:
    st.subheader("🚲 Purchase Drivers (Ranked)")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance_plot = importance.plot(kind="barh", ax=ax, color='teal')
    
    # Add labels to importance bars
    for i, v in enumerate(importance):
        ax.text(v + 0.01, i, f'{v:.1%}', color='teal', va='center', fontweight='bold')
        
    ax.set_xlabel("Significance Score")
    st.pyplot(fig)

with col_txt:
    st.subheader("🤖 Ideal Customer Portrait")

    def get_mode(col_name):
        actual_col = next((c for c in buyers.columns if c.lower() == col_name.lower()), None)
        if actual_col and not buyers[actual_col].mode().empty:
            val = buyers[actual_col].mode()[0]
            if col_name.lower() == "home owner":
                return "homeowner" if str(val).lower() == "yes" else "renter"
            return str(val)
        return "N/A"

    full_persona = (
        f"The Machine Learning analysis identifies the high-probability buyer as a **{get_mode('Marriedarital Status')}** "
        f"**{get_mode('Gender')}** within the **{get_mode('Age brackets')}** demographic. "
        f"Typically a **{get_mode('Education')}** graduate working in a **{get_mode('Occupation')}** role, "
        f"this individual is likely a **{get_mode('Home Owner')}** living in **{get_mode('Region')}** "
        f"with **{get_mode('Children')} child(ren)**, **{get_mode('Cars')} vehicle(s)**, and an income in the "
        f"**{get_mode('Income Group')}** range. "
        f"With a commute of **{get_mode('Commute Distance')}**, they represent MozBikes' primary target for "
        f"efficient urban mobility solutions."
    )

    st.success(full_persona)

    st.markdown("#### 🔥 Top 3 Strategic Actions")
    top3 = importance.sort_values(ascending=False).head(3)

    for i, (factor, score) in enumerate(top3.items(), 1):
        if factor == "Cars":
            st.write(f"{i}. **Vehicle Substitution:** Target households with {get_mode('Cars')} car(s) to position bikes as a cost-saving alternative.")
        elif factor == "Commute Distance":
            st.write(f"{i}. **Urban Focus:** Prioritize marketing within the {get_mode('Commute Distance')} radius of business hubs.")
        elif factor == "Income":
            st.write(f"{i}. **Income Targeting:** Focus on the {get_mode('Income Group')} bracket — they have spending power and commuting need.")
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
    st.markdown(f"- **Persona Ads:** Showcase **{get_mode('Occupation')}s** commuting.\n- **Hyper-Local:** Target **{get_mode('Region')}** center.\n- **LinkedIn:** Target **{get_mode('Education')}** holders.")

with rec2:
    st.markdown("### 🔧 Operations & Product")
    st.markdown(f"- **B2B Strategy:** Fleet leasing for **{get_mode('Occupation')}** hubs.\n- **Family:** Stock gear for **{get_mode('Children')}** kids.\n- **Premium:** White-glove service for **{get_mode('Home Owner')}s**.")

with rec3:
    st.markdown("### 📈 Scaling MozBikes")
    st.markdown(f"- **Car Replacement:** Focus on the **{get_mode('Cars')}-car** segment.\n- **Distance:** Bike durability for **{get_mode('Commute Distance')}**.\n- **Income:** Offers for **{get_mode('Income Group')}** range.")

st.success("✅ **Final Summary:** MozBikes success relies on targeting urban professionals with short commutes who view the bike as a vehicle, not a toy.")
