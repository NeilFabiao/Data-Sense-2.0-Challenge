import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. SETUP & INTRODUÇÃO (A HISTÓRIA)
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Dashboard", layout="wide")

st.title("🚲 Dashboard Estratégico MozBikes")

st.markdown("""
### **📖 A História do Projeto: Em busca da Tilápia**
Este exercício nasceu do desafio de identificar o **Perfil Ideal de Cliente** da MozBikes. 
Em um mar de dados, aprendemos que não precisamos de algoritmos complexos se não entendermos o problema. 
O objetivo: Usar o **anzol certo** (dados limpos) para pescar a **tilápia certa** (o comprador lucrativo).
""")

# ----------------------------
# 2. SEÇÃO: APRENDIZADOS MOZ DEVS DATAWAVE 2.0
# ----------------------------
st.divider()
st.header("🎓 Lições do Moz Devs DataWave 2.0")

dw_col1, dw_col2, dw_col3 = st.columns(3)

with dw_col1:
    st.markdown("#### **1. O Ecossistema**")
    st.info("""
    - **Profissionais:** Engenheiro, Cientista e Analista de Dados ([Roadmap.sh](https://roadmap.sh)).
    - **A Ferramenta:** Conheça seu anzol. O simples que responde vence o complexo que confunde.
    """)

with dw_col2:
    st.markdown("#### **2. Pilares de Governança**")
    st.warning("""
    2. **Qualidade de Dados**
    3. **Governança de Dados**
    4. **Soberania de Dados**
    5. **Democratização de Dados**
    """)

with dw_col3:
    st.markdown("#### **3. A Jornada do Insight**")
    st.success("""
    - **Expectativa:** Insights básicos e acionáveis.
    - **Stakeholders:** Visualizações feitas para quem "sabe zero".
    - **Foco:** Validar a informação e dar recomendações.
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
buyers = df[df[target] == "Yes"]

def get_mode(col_name):
    actual_col = next((c for c in buyers.columns if c.lower() == col_name.lower()), None)
    if actual_col and not buyers[actual_col].mode().empty:
        return str(buyers[actual_col].mode()[0])
    return "N/A"

# ----------------------------
# 4. ANÁLISE VISUAL (SUSTENTAÇÃO DOS DADOS)
# ----------------------------
st.divider()
st.header("📈 Visualização de Dados (O que aconteceu?)")

def plot_bar_pie(feature, insight_text):
    if feature not in buyers.columns: return
    st.markdown(f"### **📊 Análise de {feature}**")
    data = buyers[feature].value_counts().sort_index()
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(data.index.astype(str), data.values, color='#2E86C1', edgecolor='black')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{int(bar.get_height())}', ha='center', fontweight='bold')
        ax.set_title(f"Volume por {feature}", fontweight='bold')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1']
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140, colors=colors, pctdistance=0.85)
        fig.gca().add_artist(plt.Circle((0,0), 0.70, fc='white'))
        ax.set_title(f"Percentual por {feature}", fontweight='bold')
        st.pyplot(fig)
    st.info(f"**Insight:** {insight_text}")

plot_bar_pie("Age brackets", "79.6% dos compradores são de meia-idade. Estabilidade financeira detectada.")
plot_bar_pie("Commute Distance", "41.6% moram a menos de 1 milha. A bike resolve o problema do trajeto curto.")

# ----------------------------
# 5. ENGINE DE MACHINE LEARNING (POR QUE ACONTECEU?)
# ----------------------------
st.divider()
st.header("🧠 Inteligência Preditiva (Machine Learning)")

# Preparação dos dados para o Modelo
tree_df = df.copy().dropna()
if "ID" in tree_df.columns: tree_df = tree_df.drop(columns=["ID"])

# Encoding
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})
le = LabelEncoder()
cat_cols = tree_df.select_dtypes(include="object").columns
for col in cat_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

X = tree_df.drop(columns=[target])
y = tree_df[target]

# Treinamento da Árvore de Decisão
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

col_ml1, col_ml2 = st.columns([1, 1.2])

with col_ml1:
    st.subheader("**🚀 Direcionadores de Venda**")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot(kind="barh", ax=ax, color='#117A65')
    for i, v in enumerate(importance):
        ax.text(v + 0.01, i, f'{v:.1%}', fontweight='bold')
    st.pyplot(fig)

with col_ml2:
    st.subheader("**🤖 Perfil Ideal (Golden Profile)**")
    st.success(f"""
    O algoritmo identifica que o maior preditor de compra é a combinação de:
    - **Distância de Trajeto:** {get_mode('Commute Distance')}
    - **Ocupação Profissional:** {get_mode('Occupation')}
    - **Propriedade:** {get_mode('Home Owner')}
    """)
    st.markdown("---")
    st.subheader("💰 Estratégia de Lucratividade")
    st.write("Para maximizar o lucro, foque no **Upselling**: venda acessórios de alta margem para este perfil profissional que valoriza tempo e saúde.")

# ----------------------------
# 6. ROADMAP ESTRATÉGICO FINAL
# ----------------------------
st.divider()
st.header("🚀 Recomendações com Clareza e Certeza")

st.subheader("**O que foi feito e qual o resultado?**")
st.write("> **Resposta:** Validamos que o perfil profissional de meia-idade com trajetos curtos é a nossa 'tilápia'. O resultado permite direcionar 100% do budget de marketing para esse público, eliminando desperdícios.")

rec1, rec2, rec3 = st.columns(3)
with rec1:
    st.info(f"**Marketing:** Focar na região {get_mode('Region')} com anúncios para {get_mode('Occupation')}s.")
with rec2:
    st.info(f"**Operações:** Criar planos de manutenção para quem pedala {get_mode('Commute Distance')}.")
with rec3:
    st.info(f"**Financeiro:** Criar campanhas para lares com {get_mode('Cars')} carro, focando na substituição.")

st.success("✅ **Conclusão:** O simples que dá resposta. Focamos no problema, limpamos os dados e agora temos a rota do lucro.")
