import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# SETUP & CONFIG
# ----------------------------
st.set_page_config(page_title="MozBikes Analysis Dashboard", layout="wide")

st.title("🚲 MozBikes Strategic Analysis")
st.markdown("""
Esta análise foca no perfil de clientes que **compraram** uma bicicleta, utilizando Machine Learning para identificar os principais motivadores de venda e gerar recomendações estratégicas automáticas.
""")

# Carregamento de dados
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

df.columns = df.columns.str.strip()
target = "Purchased Bike"

# Filtrar apenas compradores para análise demográfica
buyers = df[df[target] == "Yes"]

# ----------------------------
# FUNÇÃO: GRÁFICOS + RESUMO
# ----------------------------
def plot_bar_pie(feature, summary_text=""):
    st.divider()
    st.subheader(f"📊 {feature} - Perfil dos Compradores")
    
    data = buyers[feature].value_counts()
    col1, col2 = st.columns(2)

    # BAR CHART
    with col1:
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax, color='#1f77b4')
        ax.set_ylabel("Quantidade")
        ax.set_title(f"Distribuição por {feature}")
        st.pyplot(fig)

    # PIE CHART
    with col2:
        fig, ax = plt.subplots()
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90, 
               colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_title(f"Percentual por {feature}")
        ax.axis("equal")
        st.pyplot(fig)
    
    if summary_text:
        st.info(summary_text)

# ----------------------------
# SEÇÕES DE ANÁLISE DEMOGRÁFICA
# ----------------------------
st.header("📈 Perfil do Consumidor")

if "Gender" in df.columns:
    plot_bar_pie("Gender", "**Insight:** Mercado equilibrado entre gêneros (aprox. 50/50), indicando marketing universal.")

if "Education" in df.columns:
    plot_bar_pie("Education", "**Insight:** ~79.3% dos compradores possuem ensino superior, indicando um público instruído.")

if "Occupation" in df.columns:
    plot_bar_pie("Occupation", "**Insight:** Profissionais e técnicos lideram. Trabalhadores manuais são o menor nicho (11.4%).")

if "Commute Distance" in df.columns:
    plot_bar_pie("Commute Distance", "**Insight:** 57% dos compradores percorrem menos de 2 milhas. O uso é estritamente urbano/local.")

if "Age brackets" in df.columns:
    plot_bar_pie("Age brackets", "**Insight:** Meia-idade (Middle Age) domina com 79.6% das compras.")

# ----------------------------
# MACHINE LEARNING SECTION
# ----------------------------
st.divider()
st.title("🌳 Machine Learning: Decision Tree Insights")

# 1. Preparação de Dados para ML
tree_df = df.copy().dropna()
if 'ID' in tree_df.columns:
    tree_df = tree_df.drop(columns=['ID'])

# 2. Encoding
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns
for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

# 3. Model Training
X = tree_df.drop(columns=[target])
y = tree_df[target]
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X, y)

# 4. Feature Importance Calculation
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance[importance > 0].sort_values(ascending=True)

# ----------------------------
# 🤖 AUTOMATED SUMMARY GENERATOR
# ----------------------------
col_chart, col_ai = st.columns([1, 1])

with col_chart:
    st.subheader("🚲 Key Drivers of Purchase")
    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax, color='teal')
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

with col_ai:
    st.subheader("🤖 AI Business Summary")
    
    # Extract data for the summary
    top_drivers = importance.sort_values(ascending=False)
    primary_feat = top_drivers.index[0]
    secondary_feat = top_drivers.index[1]
    top_score = top_drivers.values[0]

    # Dynamic summary logic
    summary_text = f"""
    O modelo de Machine Learning identificou que **{primary_feat}** é o fator determinante, 
    com um peso de **{top_score:.1%}** na decisão de compra. 
    
    O segundo fator mais relevante é **{secondary_feat}**.
    """
    
    if primary_feat == "Cars":
        summary_text += "\n\n**Análise:** O interesse por bicicletas está diretamente ligado à posse de veículos. Clientes com menos carros são seu alvo principal."
    elif primary_feat == "Income":
        summary_text += "\n\n**Análise:** O poder aquisitivo é a maior barreira ou motor de vendas atual."
    
    st.success(summary_text)

# ----------------------------
# STRATEGIC RECOMMENDATIONS
# ----------------------------
st.divider()
st.title("📌 Strategic Recommendations for MozBikes")

# Logic-based Priority
st.subheader(f"🚀 Priority Action based on {primary_feat}")
if primary_feat == "Cars":
    st.write("👉 **Strategy:** Launch a 'Swap your Second Car for a Bike' campaign. Focus on the savings in fuel and maintenance for households with multiple members but only 1 car.")
elif "Distance" in primary_feat:
    st.write("👉 **Strategy:** Partner with city planners and office hubs to create 'Last Mile' bike stations within 1 mile of major workplaces.")
else:
    st.write(f"👉 **Strategy:** Prioritize resource allocation based on {primary_feat} trends to optimize inventory.")

# Compiled Recommendations
col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.markdown("""
    ### 🎯 Marketing & Sales
    * **Target the 'Middle-Aged Professional':** Focus on health and efficiency rather than sports/adventure.
    * **Education-Led Outreach:** University graduates are your best customers. Partner with Alumni networks and Tech parks.
    * **Urban Mobility Focus:** 57% travel < 2 miles. Market the bike as a 'Traffic Buster' for short trips.
    """)

with col_rec2:
    st.markdown("""
    ### ⚙️ Product & Operations
    * **Car-Free Incentives:** Target areas with low car density but high professional employment.
    * **Corporate Mobility:** Since 31% are Professionals, offer B2B fleet leasing to companies for employee transport.
    * **Inclusive Design:** Maintain neutral branding, as the buyer split is perfectly balanced between men and women.
    """)

# Final Executive Wrap-up
st.divider()
st.info(f"""
**Final Strategy Note:** MozBikes should stop seeing itself as a bike retailer and start acting as an **Urban Mobility Partner**. 
The data proves the primary customer is an **educated professional** looking for an alternative to **car transport** for **short distances**.
""")
