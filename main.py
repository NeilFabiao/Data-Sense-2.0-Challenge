import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. CONFIGURAÇÃO & TEMA
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Dashboard", layout="wide")

st.title("🚲 Dashboard Estratégico MozBikes")
st.markdown("### Inteligência de Dados aplicada à Mobilidade Urbana")

# ----------------------------
# 2. SEÇÃO: APRENDIZADOS MOZ DEVS DATAWAVE 2.0
# ----------------------------
st.divider()
st.header("🎓 Lições do Moz Devs DataWave 2.0")

# Criando as 3 Seções de Aprendizado
dw_col1, dw_col2, dw_col3 = st.columns(3)

with dw_col1:
    st.markdown("#### **1. O Ecossistema e a Ferramenta**")
    st.info("""
    - **Profissionais:** Engenheiro, Cientista e Analista de Dados ([Roadmap.sh](https://roadmap.sh)).
    - **A Analogia da Tilápia:** "Se quer pescar tilápia, conheça sua ferramenta. Às vezes não precisa de um barco tecnológico, um **anzol simples** é o suficiente."
    - **Evolução:** Saímos da falta de dados para a abundância; agora o foco é a mineração inteligente.
    """)

with dw_col2:
    st.markdown("#### **2. Pilares da Maturidade**")
    st.warning("""
    2. **Qualidade de Dados:** A base de tudo.
    3. **Governança de Dados:** Regras e processos.
    4. **Soberania de Dados:** Controle e segurança.
    5. **Democratização:** Dados acessíveis para todos.
    """)

with dw_col3:
    st.markdown("#### **3. Sua Jornada e Primeiro Projeto**")
    st.success("""
    - **O que esperam ver:** Insights básicos, mas acionáveis.
    - **Simplicidade:** O simples que dá resposta é melhor que o complexo que confunde.
    - **Stakeholders:** Faça para quem "sabe zero"; use labels claras e explicações.
    """)

# Seção de Metodologia de Resolução
with st.expander("🛠️ Metodologia: Como resolvemos o problema", expanded=False):
    st.markdown("""
    1. **Definir o Problema:** Qual a dor de negócio?
    2. **Medir (KPIs):** Definir métricas de sucesso.
    3. **Limpeza de Dados:** Validar se o dataset está correto.
    4. **Modelos Preditivos:** Antecipar tendências.
    5. **Resultado:** Não dar apenas a resposta, mas dar a **recomendação**.
    """)

# ----------------------------
# 3. CARREGAMENTO E LIMPEZA
# ----------------------------
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

target = "Purchased Bike"

# Binning de Renda para Stakeholders
df["Faixa de Renda"] = pd.cut(
    df["Income"],
    bins=[0, 30000, 60000, 90000, 120000, float("inf")],
    labels=["Até 30k", "30k–60k", "60k–90k", "90k–120k", "120k+"]
)

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
# 4. SUMÁRIO EXECUTIVO (KPIs)
# ----------------------------
st.divider()
st.subheader("**🎯 O que foi feito e qual o Resultado (KPIs Chave)**")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Perfil de Idade", get_mode("Age brackets"), help="Faixa etária com maior conversão.")
m2.metric("Região Crítica", get_mode("Region"), help="Localidade onde a demanda é maior.")
m3.metric("Uso Principal", "Trajetos Curtos", help="Baseado na Commute Distance.")
m4.metric("Persona Líder", get_mode("Occupation"), help="Segmento profissional dominante.")

# ----------------------------
# 5. ANÁLISE VISUAL (O QUE OS DADOS DIZEM?)
# ----------------------------
st.header("📈 Análise de Perfil (O simples que responde)")

def plot_bar_pie(feature, insight_text):
    if feature not in buyers.columns: return

    st.divider()
    st.markdown(f"### **📊 Análise de {feature}**")
    
    data = buyers[feature].value_counts().sort_index()
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(data.index.astype(str), data.values, color='#2E86C1', edgecolor='black')
        ax.set_title(f"QUANTIDADE POR {feature.upper()}", fontweight='bold')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{int(bar.get_height())}', ha='center', fontweight='bold')
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1']
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140, colors=colors, pctdistance=0.85)
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.set_title(f"PERCENTUAL: {feature.upper()}", fontweight='bold')
        st.pyplot(fig)

    st.info(f"**Recomendação:** {insight_text}")

# Chamadas com insights diretos
plot_bar_pie("Occupation", "Focar marketing em profissionais e técnicos; eles buscam eficiência no deslocamento para o trabalho.")
plot_bar_pie("Commute Distance", "A maioria mora perto do trabalho (0-1 milhas). A bike deve ser vendida como 'mais rápida que o trânsito'.")
plot_bar_pie("Cars", "Pessoas com 1 ou nenhum carro são o alvo. Venda a ideia da 'liberdade de não precisar de um segundo carro'.")

# ----------------------------
# 6. MODELO PREDITIVO
# ----------------------------
st.divider()
st.header("🧠 Inteligência Preditiva")

tree_df = df.copy().dropna()
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})
le = LabelEncoder()
cat_cols = tree_df.select_dtypes(include="object").columns
for col in cat_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

X = tree_df.drop(columns=[target, "Faixa de Renda"])
y = tree_df[target]

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

col_ml1, col_ml2 = st.columns([1, 1.2])

with col_ml1:
    st.subheader("**🚀 O que mais impacta a compra?**")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot(kind="barh", ax=ax, color='#117A65')
    for i, v in enumerate(importance):
        ax.text(v + 0.01, i, f'{v:.1%}', color='black', va='center', fontweight='bold')
    st.pyplot(fig)

with col_ml2:
    st.subheader("**🤖 Perfil Ideal (Golden Profile)**")
    st.success(f"""
    O cliente MozBikes com maior chance de conversão é um **{get_mode('Marital Status')}**, 
    do gênero **{get_mode('Gender')}**, na faixa de **{get_mode('Age brackets')}**. 
    Este perfil vive em **{get_mode('Region')}**, possui nível superior e trabalha em áreas de **{get_mode('Occupation')}**.
    """)
    st.markdown(f"👉 **Insight de Ouro:** O trajeto de **{get_mode('Commute Distance')}** é o maior influenciador da decisão.")

# ----------------------------
# 7. ROADMAP ESTRATÉGICO
# ----------------------------
st.divider()
st.header("🚀 Roadmap Estratégico MozBikes")
rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown("### 📢 Aquisição de Mercado")
    st.markdown(f"""
    - **Ads de Persona:** Use imagens de **{get_mode('Occupation')}s** a caminho do trabalho.
    - **Foco Local:** Campanhas direcionadas para a região de **{get_mode('Region')}**.
    """)

with rec2:
    st.markdown("### 🔧 Operações e Produto")
    st.markdown(f"""
    - **Parceria B2B:** Oferecer planos para empresas com muitos funcionários técnicos.
    - **Kits Família:** Acessórios para pais com **{get_mode('Children')}** filhos.
    """)

with rec3:
    st.markdown("### 📈 Expansão")
    st.markdown(f"""
    - **Substituição de Carros:** Focar em quem tem apenas **{get_mode('Cars')}** carro.
    - **Financiamento:** Facilitar para a faixa de **Até 60k** de renda.
    """)

st.success("✅ **Conclusão:** MozBikes deve focar na tilápia (profissional urbano) usando o anzol certo (mobilidade eficiente).")
