import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# SETUP
# ----------------------------
st.set_page_config(page_title="Análise Bike Buyers", layout="wide")

st.title("🚲 Bike Buyers Analysis (YES Only)")
st.markdown("Esta análise foca exclusivamente no perfil de clientes que **compraram** uma bicicleta, identificando padrões demográficos e comportamentais.")

# Carregamento de dados (ajuste o caminho se necessário)
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
except:
    st.error("Arquivo não encontrado. Verifique o nome do arquivo Excel.")
    st.stop()

df.columns = df.columns.str.strip()
target = "Purchased Bike"

# Filtrar apenas compradores
buyers = df[df[target] == "Yes"]

# ----------------------------
# FUNÇÃO: GRÁFICOS + RESUMO
# ----------------------------
def plot_bar_pie(feature, summary_text=""):
    st.divider()
    st.subheader(f"📊 {feature} - Perfil dos Compradores")
    
    data = buyers[feature].value_counts()
    col1, col2 = st.columns(2)

    # ---------------- BAR CHART ----------------
    with col1:
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax, color='#1f77b4')
        ax.set_ylabel("Quantidade")
        ax.set_title(f"Distribuição por {feature}")
        st.pyplot(fig)

    # ---------------- PIE CHART ----------------
    with col2:
        fig, ax = plt.subplots()
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_title(f"Percentual por {feature}")
        ax.axis("equal")
        st.pyplot(fig)
    
    # Exibir o texto de resumo logo abaixo dos gráficos
    if summary_text:
        st.info(summary_text)

# ----------------------------
# SEÇÕES DE ANÁLISE COM INSIGHTS
# ----------------------------

if "Gender" in df.columns:
    resumo_genero = "**Insight:** O mercado está perfeitamente equilibrado entre homens (50.3%) e mulheres (49.7%), indicando que o gênero não é um fator limitante para vendas."
    plot_bar_pie("Gender", resumo_genero)

if "Marital Status" in df.columns:
    resumo_civil = "**Insight:** Analise se clientes casados ou solteiros têm maior propensão ao lazer ou transporte prático."
    plot_bar_pie("Marital Status", resumo_civil)

if "Education" in df.columns:
    resumo_edu = "**Insight:** Cerca de 79.3% dos compradores possuem ensino superior (completo ou incompleto), sugerindo um público com maior nível de instrução."
    plot_bar_pie("Education", resumo_edu)

if "Occupation" in df.columns:
    resumo_occ = "**Insight:** Profissionais liberais e técnicos especializados lideram as compras. Trabalhadores manuais representam a menor fatia (11.4%)."
    plot_bar_pie("Occupation", resumo_occ)

if "Region" in df.columns:
    resumo_reg = "**Insight:** A América do Norte é o principal mercado (45.7%), seguida pela Europa. O foco de marketing deve priorizar estas regiões."
    plot_bar_pie("Region", resumo_reg)

if "Commute Distance" in df.columns:
    resumo_dist = "**Insight:** Uso predominantemente urbano e de curta distância. 57% dos compradores percorrem menos de 2 milhas."
    plot_bar_pie("Commute Distance", resumo_dist)

if "Age brackets" in df.columns:
    resumo_idade = "**Insight:** A meia-idade domina esmagadoramente com 79.6% das compras. Adolescentes e idosos são nichos muito menores."
    plot_bar_pie("Age brackets", resumo_idade)

# ----------------------------
# MACHINE LEARNING SECTION (IMPROVED)
# ----------------------------
st.divider()
st.title("🌳 What Drives Bike Purchases (Decision Tree)")

# 1. Clean and Prepare Data
tree_df = df.copy().dropna()

# --- THE FIX: Drop irrelevant columns like 'ID' ---
# Also drop columns that might be 'leakage' or redundant
cols_to_drop = ['ID'] 
# Add any other columns here that don't make sense as predictors
tree_df = tree_df.drop(columns=[c for c in cols_to_drop if c in tree_df.columns])

# 2. Encode Target
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})

# 3. Encode Categorical Features
# We use a dictionary to keep track of encoders if we wanted to predict later
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns

for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

# 4. Define Features (X) and Target (y)
X = tree_df.drop(columns=[target])
y = tree_df[target]

# 5. Train Model
# We set min_samples_leaf to prevent the model from picking 
# up "noise" or outliers like specific car counts that don't generalize.
model = DecisionTreeClassifier(
    max_depth=4, 
    min_samples_leaf=10, 
    random_state=42
)
model.fit(X, y)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
st.subheader("🚲 Key Drivers of Bike Purchase")
st.write("This chart shows which factors (Age, Income, Region, etc.) actually influence the decision to buy.")

importance = pd.Series(model.feature_importances_, index=X.columns)
# Filter out features with 0 importance to keep the chart clean
importance = importance[importance > 0].sort_values()

fig, ax = plt.subplots()
importance.plot(kind="barh", ax=ax, color='teal')
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# ----------------------------
# TOP FACTORS TABLE
# ----------------------------
st.subheader("🔥 Top Factors Explained")

top_factors = importance.sort_values(ascending=False).head(5)

if not top_factors.empty:
    for factor, score in top_factors.items():
        if factor == "Cars":
            st.write(f"**{factor}:** Usually the strongest driver. People with fewer cars tend to buy more bikes.")
        elif factor == "Commute Distance":
            st.write(f"**{factor}:** Distance to work significantly changes the need for a bike.")
        else:
            st.write(f"**{factor}:** This feature has a {score:.2%} impact on the decision.")
else:
    st.write("No significant drivers found. Check if the dataset has enough variation.")


# ----------------------------
# RECOMMENDATIONS SECTION (DATA-DRIVEN)
# ----------------------------
st.divider()
st.title("📌 Strategic Recommendations for MozBikes")

st.markdown("""
Based on the buyer profile analysis and machine learning insights, the following strategies are recommended to maximize growth and market penetration.
""")

# ----------------------------
# CORE STRATEGY
# ----------------------------
st.subheader("🚲 1. Position Bikes as a Daily Transport Solution (NOT Leisure)")
st.markdown("""
The data shows that buyers are primarily **working professionals with structured lifestyles**.

👉 MozBikes should position bicycles as:
- A **reliable commuting tool**
- A **cost-efficient alternative to cars**
- A **solution to urban traffic challenges**

❗ Avoid marketing bikes purely as recreational products.
""")

# ----------------------------
# CUSTOMER SEGMENT
# ----------------------------
st.subheader("🎯 2. Focus on Educated Working Professionals")
st.markdown("""
- ~60% of buyers have **college or university education**
- Top occupations: **Professional (31%) & Skilled Manual (24%)**

👉 Target:
- Office workers
- Technicians
- Skilled labor workforce

💡 Strategy:
- Partner with companies
- Offer **employee mobility programs**
- Provide **corporate discounts**
""")

# ----------------------------
# URBAN MOBILITY
# ----------------------------
st.subheader("🏙️ 3. Double Down on Short-Distance Urban Commuters")
st.markdown("""
Buyers are primarily **short-distance commuters**.

👉 MozBikes should:
- Focus on **city users**
- Promote:
  - Faster commute times
  - Avoiding traffic
  - Lower transport costs

💡 Opportunity:
- Position bikes as a **"last-mile solution"**
""")

# ----------------------------
# VEHICLE SUBSTITUTION
# ----------------------------
st.subheader("🚗 4. Compete with Cars, Not Other Bikes")
st.markdown("""
The decision tree indicates that **car ownership impacts bike purchases**.

👉 Insight:
- People are not choosing between bikes…
- They are choosing between **cars vs bikes**

💡 Strategy:
- Highlight:
  - Fuel savings
  - Maintenance savings
  - Parking convenience
""")

# ----------------------------
# GENDER STRATEGY
# ----------------------------
st.subheader("🚻 5. Maintain a Fully Inclusive Strategy")
st.markdown("""
The gender split is **50/50**.

👉 Implication:
- No need for gender-specific targeting
- Focus on **universal value propositions**

💡 Avoid:
- Over-segmentation by gender
""")

# ----------------------------
# PRODUCT STRATEGY
# ----------------------------
st.subheader("💰 6. Build a Commuter-Focused Product Line")
st.markdown("""
Design products specifically for daily usage:

👉 Recommended offerings:
- Durable commuter bikes
- Low-maintenance models
- Affordable entry-level options

💡 Add-ons:
- Baskets / storage
- Safety features (lights, reflectors)
""")

# ----------------------------
# GO-TO-MARKET
# ----------------------------
st.subheader("📣 7. Smart Go-To-Market Strategy")
st.markdown("""
👉 Focus marketing on:
- Urban professionals
- Daily commuters

Channels:
- Workplace partnerships
- Digital campaigns targeting working adults

Messaging:
- “Save money on transport”
- “Beat traffic”
- “Arrive faster”
""")

# ----------------------------
# DATA STRATEGY
# ----------------------------
st.subheader("🤖 8. Use Data as a Competitive Advantage")
st.markdown("""
The decision tree highlights key purchase drivers.

👉 MozBikes should:
- Continuously update models
- Identify **high-probability buyers**
- Run targeted campaigns

💡 Future step:
- Build a **recommendation engine for customers**
""")

# ----------------------------
# FINAL EXECUTIVE SUMMARY
# ----------------------------
st.success("""
✅ **Key Takeaway:**  
MozBikes is not in the “bike market” — it is in the **urban mobility market**.

Success will come from targeting **working professionals with short commutes**, positioning bikes as a **practical alternative to cars**, and using **data-driven strategies** to scale efficiently.
""")
