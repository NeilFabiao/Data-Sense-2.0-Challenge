import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. SETUP & IDENTIDADE VISUAL
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Analysis", layout="wide")

st.title("🚲 Dashboard Estratégico MozBikes")

# História e Contexto do Projeto
st.markdown("""
### **📖 Contexto do Desafio: A Busca pelo Perfil Ideal**
Este projeto foi desenvolvido como um exercício estratégico para a **MozBikes**. O objetivo central foi identificar, através de variáveis demográficas e socioeconômicas, qual é o perfil de cliente com maior probabilidade de comprar uma bicicleta.

Como uma **tarefa oculta**, o desafio exigia que as conclusões fossem sustentadas por visualizações de dados claras, fornecendo recomendações com clareza e certeza para os stakeholders. 

#### **🎯 O Perfil "Golden": O Profissional Urbano de Meia-Idade**
Com base nos dados analisados, o perfil com maior probabilidade de conversão é composto por:
* **Idade:** Indivíduos de meia-idade (**79,6%** dos compradores).
* **Ocupação:** Profissionais (**31,2%**).
* **Logística:** Pessoas que vivem a **0-1 milha** do trabalho (**41,6%**).
* **Geografia:** Residentes da **América do Norte** (**45,7%**).
* **Educação:** Graduados com nível de **Bacharelado** (**35,1%**).

**Justificativa Econômica e Comportamental:** Este grupo possui estabilidade financeira e alto poder aquisitivo. Comportamentalmente, a compra é motivada pela **eficiência**: para quem mora a menos de 1 milha do trabalho, a bicicleta é a solução de mobilidade mais lógica e rápida. Além disso, a quantidade de **carros** é o fator de maior influência na decisão, indicando que a bike é vista como um substituto estratégico ao veículo motorizado.
"""""")

# ----------------------------
# 2. SEÇÃO: APRENDIZADOS MOZ DEVS DATAWAVE 2.0
# ----------------------------
st.divider()
st.header("🎓 Lições do Moz Devs DataWave 2.0")

dw_col1, dw_col2, dw_col3 = st.columns(3)

with dw_col1:
    st.markdown("#### **1. O Ecossistema e a Ferramenta**")
    st.info("""
    - **Profissionais:** Engenheiro, Cientista e Analista de Dados ([Roadmap.sh](https://roadmap.sh)).
    - **A Analogia:** Se quer pescar tilápia, conheça sua ferramenta. Às vezes um **anzol simples** é o suficiente.
    - **Mineração:** Saímos da falta de dados para a abundância; o foco agora é a extração de valor.
    """)

with dw_col2:
    st.markdown("#### **2. Pilares da Maturidade**")
    st.warning("""
    2. **Qualidade de Dados**
    3. **Governança de Dados**
    4. **Soberania de Dados**
    5. **Democratização de Dados**
    """)

with dw_col3:
    st.markdown("#### **3. A Jornada do Insight**")
    st.success("""
    - **O que esperam:** Insights básicos que respondam o problema.
    - **Simplicidade:** O simples que dá resposta ganha do complexo que confunde.
    - **Ação:** Não dê apenas a resposta, dê a **recomendação estratégica**.
    """)

# ----------------------------
# 3. CARREGAMENTO E TRATAMENTO (LIMPEZA)
# ----------------------------
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

target = "Purchased Bike"

# Binning de Renda para o Stakeholder (Simplificação)
df["Income Group"] = pd.cut(
    df["Income"],
    bins=[0, 30000, 60000, 90000, 120000, float("inf")],
    labels=["<30k", "30k–60k", "60k–90k", "90k–120k", "120k+"]
)

# Filtro apenas para compradores (Análise de Perfil de Sucesso)
buyers = df[df[target] == "Yes"]

def get_mode(col_name):
    actual_col = next((c for c in buyers.columns if c.lower() == col_name.lower()), None)
    if actual_col and not buyers[actual_col].mode().empty:
        val = buyers[actual_col].mode()[0]
        if col_name.lower() == "home owner":
            return "Proprietário" if str(val).lower() == "yes" else "Inquilino"
        return str(val)
    return "N/A"

# ----------------------------
# 4. ANÁLISE VISUAL (O QUE OS DADOS DIZEM)
# ----------------------------
st.divider()
st.header("📈 Visualização de Dados (Sustentação das Conclusões)")

def plot_bar_pie(feature, summary_text=""):
    if feature not in buyers.columns: return

    st.markdown(f"### **📊 Análise de {feature}**")
    data = buyers[feature].value_counts().sort_index()
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(data.index.astype(str), data.values, color='#2E86C1', edgecolor='black')
        ax.set_title(f"Volume por {feature.upper()}", fontweight='bold')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{int(bar.get_height())}', ha='center', fontweight='bold')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1']
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140, colors=colors, pctdistance=0.85)
        fig.gca().add_artist(plt.Circle((0,0), 0.70, fc='white'))
        ax.set_title(f"Distribuição de {feature.upper()}", fontweight='bold')
        st.pyplot(fig)

    if summary_text: st.info(f"**Insight:** {summary_text}")

# Análise de todas as variáveis chave
plot_bar_pie("Age brackets", "Meia-idade domina 79.6% das compras. Focar marketing neste público estável.")
plot_bar_pie("Commute Distance", "41.6% moram a menos de 1 milha. A bike é a solução logística ideal.")
plot_bar_pie("Occupation", "Profissionais e técnicos são o motor de vendas.")
plot_bar_pie("Cars", "Quem possui menos carros tem maior propensão a comprar para substituir o segundo veículo.")
plot_bar_pie("Income Group", "Público de renda média-alta (60k-90k) é o mais lucrativo.")

# ----------------------------
# 5. ENGINE DE MACHINE LEARNING (POR QUE ACONTECEU?)
# ----------------------------
st.divider()
st.header("🧠 Inteligência de Dados: O 'Golden Profile'")

tree_df = df.copy().dropna()
if "ID" in tree_df.columns: tree_df = tree_df.drop(columns=["ID"])

# Encoding para ML usando todas as variáveis
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns
for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

X = tree_df.drop(columns=[target, "Income Group"]) # Drop apenas o derivado
y = tree_df[target]

model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance[importance > 0].sort_values()

col_ml1, col_ml2 = st.columns([1, 1.2])

with col_ml1:
    st.subheader("**🚀 Direcionadores de Decisão**")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot(kind="barh", ax=ax, color='#117A65')
    for i, v in enumerate(importance):
        ax.text(v + 0.005, i, f'{v:.1%}', fontweight='bold')
    st.pyplot(fig)

with col_ml2:
    st.subheader("**🤖 Retrato do Cliente Ideal**")
    full_persona = (
        f"O modelo identifica o comprador de alta probabilidade como um **{get_mode('Marriedarital Status')}** "
        f"do gênero **{get_mode('Gender')}** na faixa de **{get_mode('Age brackets')}**. "
        f"Geralmente um profissional de nível **{get_mode('Education')}** que trabalha como **{get_mode('Occupation')}**. "
        f"Ele é **{get_mode('Home Owner')}**, reside na região **{get_mode('Region')}**, tem **{get_mode('Children')} filho(s)** "
        f"e possui **{get_mode('Cars')} carro(s)**. O trajeto de **{get_mode('Commute Distance')}** é o gatilho da venda."
    )
    st.success(full_persona)
    
    st.markdown("#### **💰 Estratégia de Lucro**")
    st.write("1. **Venda de Valor:** Foque na economia de combustível para profissionais.")
    st.write("2. **Upselling:** Este perfil valoriza acessórios de qualidade e segurança.")

# ----------------------------
# 6. ROADMAP ESTRATÉGICO FINAL (RECOMENDAÇÕES)
# ----------------------------
st.divider()
st.header("🚀 Roadmap Estratégico MozBikes")

rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown("### 📢 Aquisição de Mercado")
    st.markdown(f"""
    - **Ads Persona:** Use imagens de **{get_mode('Occupation')}s** no trajeto diário.
    - **Geofencing:** Focar 80% do budget na região de **{get_mode('Region')}**.
    - **LinkedIn:** Canal prioritário para atingir graduados em **{get_mode('Education')}**.
    """)

with rec2:
    st.markdown("### 🔧 Operações e Produto")
    st.markdown(f"""
    - **B2B:** Planos corporativos para empresas com técnicos e profissionais.
    - **Kits Família:** Estocar acessórios para quem tem **{get_mode('Children')}** filhos.
    - **Serviço Premium:** Montagem em domicílio para **{get_mode('Home Owner')}s**.
    """)

with rec3:
    st.markdown("### 📈 Expansão e Escala")
    st.markdown(f"""
    - **Substituição:** Campanhas para donos de apenas **{get_mode('Cars')}** carro.
    - **Eficiência:** Otimizar durabilidade para trajetos de **{get_mode('Commute Distance')}**.
    - **Financiamento:** Facilitar para a faixa de renda **30k-60k**.
    """)

st.success("""
✅ **Resumo Executivo Final:** O sucesso da MozBikes está na interseção da estabilidade profissional com o deslocamento urbano de curta distância. 
O modelo recomenda focar na **mobilidade profissional funcional** em vez de apenas lazer.
""")
