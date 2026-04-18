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
# MACHINE LEARNING SECTION
# ----------------------------
st.divider()
st.title("🌳 O que realmente impulsiona a compra?")
st.markdown("Utilizamos um modelo de **Árvore de Decisão** para identificar quais variáveis têm maior peso na decisão final de compra (comparando compradores vs. não compradores).")

tree_df = df.copy().dropna()
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})

# Codificação
categorical_cols = tree_df.select_dtypes(include="object").columns
for col in categorical_cols:
    tree_df[col] = LabelEncoder().fit_transform(tree_df[col])

X = tree_df.drop(columns=[target])
y = tree_df[target]

# Modelo
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Importância das Variáveis
st.subheader("🚲 Variáveis mais influentes (Key Drivers)")

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

fig, ax = plt.subplots()
importance.plot(kind="barh", ax=ax, color='teal')
ax.set_title("Importância das Características no Modelo")
st.pyplot(fig)

# Top 5
st.subheader("🔥 Top 5 Fatores Decisivos")
top_5 = importance.sort_values(ascending=False).head(5)
st.table(top_5)

st.write("---")
st.caption("Dashboard gerado para análise de comportamento do consumidor de bicicletas.")
