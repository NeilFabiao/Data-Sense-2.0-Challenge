import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. CONFIGURAÇÃO & TEMA
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Dashboard", layout="wide")

# Estilização para títulos em negrito e limpos
st.title("🚲 Dashboard Estratégico MozBikes")
st.markdown("### Inteligência de Dados aplicada à Mobilidade Urbana")

# ----------------------------
# 2. SEÇÃO: APRENDIZADO MOZ DEVS DATAWAVE 2.0
# ----------------------------
with st.expander("🎓 O que aprendi no Moz Devs DataWave 2.0", expanded=False):
    st.markdown("""
    **Metodologia de Resolução de Problemas:**
    1. **Definição do Problema:** Identificar claramente o que precisa ser resolvido antes de tocar nos dados.
    2. **Mensuração via KPIs:** O que não se mede, não se gere. Definir indicadores de sucesso.
    3. **Foco no Stakeholder:** Criar dashboards fáceis de ler, com labels claras e explicações simples.
    4. **Limpeza e Modelagem:** Dados limpos geram modelos preditivos confiáveis.
    
    **Pilares da Profissionalização de Dados:**
    * **Qualidade e Governança:** Garantir que o dado é a "única fonte da verdade".
    * **Soberania e Democratização:** O dado deve estar acessível a quem precisa decidir.
    * **Carreiras:** O ecossistema exige Engenheiros, Cientistas e Analistas de Dados trabalhando em conjunto (Ref: [Roadmap.sh](https://roadmap.sh)).
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

# Criando grupos de renda para facilitar a leitura do stakeholder
df["Faixa de Renda"] = pd.cut(
    df["Income"],
    bins=[0, 30000, 60000, 90000, 120000, float("inf")],
    labels=["Até 30k", "30k–60k", "60k–90k", "90k–120k", "120k+"]
)

# Filtrando apenas quem comprou para análise de perfil
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
st.subheader("**🎯 Resumo do Perfil Ideal (KPIs Chave)**")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Faixa Etária Principal", get_mode("Age brackets"))
m2.metric("Região de Foco", get_mode("Region"))
m3.metric("Distância Comum", get_mode("Commute Distance"))
m4.metric("Profissão Líder", get_mode("Occupation"))

# ----------------------------
# 5. ANÁLISE VISUAL (O QUE ACONTECEU?)
# ----------------------------
st.header("📈 Análise do Perfil do Comprador")

def plot_bar_pie(feature, insight_text):
    if feature not in buyers.columns: return

    st.divider()
    st.markdown(f"### **📊 Análise de {feature}**")
    
    data = buyers[feature].value_counts().sort_index()
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(data.index.astype(str), data.values, color='#2E86C1', edgecolor='black')
        ax.set_title(f"VOLUME POR {feature.upper()}", fontweight='bold')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{int(bar.get_height())}', ha='center', fontweight='bold')
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1']
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140, colors=colors, pctdistance=0.85)
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.set_title(f"DIVISÃO PERCENTUAL: {feature.upper()}", fontweight='bold')
        st.pyplot(fig)

    st.info(f"**Insight Profissional:** {insight_text}")

# Chamadas das análises
plot_bar_pie("Gender", "Equilíbrio entre gêneros indica que a comunicação não deve ser segmentada por sexo.")
plot_bar_pie("Occupation", "Profissionais e técnicos são a maior fatia; foque em parcerias corporativas.")
plot_bar_pie("Commute Distance", "A maioria percorre distâncias curtas. A bike é vista como solução de agilidade urbana.")
plot_bar_pie("Cars", "Quem possui menos carros tem maior propensão à compra. Bike como substituto do 2º carro.")

# ----------------------------
# 6. MODELO PREDITIVO (POR QUE ACONTECEU?)
# ----------------------------
st.divider()
st.header("🧠 Inteligência de Dados e Predição")

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
    st.subheader("**🚀 Principais Direcionadores de Venda**")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot(kind="barh", ax=ax, color='#117A65')
    ax.set_title("IMPACTO DAS VARIÁVEIS NA DECISÃO", fontweight='bold')
    st.pyplot(fig)

with col_ml2:
    st.subheader("**🤖 Perfil 'Golden' MozBikes**")
    full_persona = (
        f"Com base no modelo de Machine Learning, o cliente ideal é **{get_mode('Marital Status')}**, "
        f"do gênero **{get_mode('Gender')}**, na faixa de **{get_mode('Age brackets')}**. "
        f"Geralmente é um **{get_mode('Occupation')}** com nível superior (**{get_mode('Education')}**), "
        f"**{get_mode('Home Owner')}** e vive na região de **{get_mode('Region')}**. "
        f"Possui **{get_mode('Children')} filho(s)** e **{get_mode('Cars')} carro(s)**. "
        f"Seu trajeto de **{get_mode('Commute Distance')}** é o gatilho principal para escolher a MozBikes."
    )
    st.success(full_persona)
    
    st.markdown("#### **🔥 Ações Recomendadas (Top 3)**")
    top3 = importance.sort_values(ascending=False).head(3)
    for i, (fator, score) in enumerate(top3.items(), 1):
        st.write(f"{i}. **{fator}:** Otimizar campanhas focadas em quem possui **{get_mode(fator)}**.")

# ----------------------------
# 7. ROADMAP ESTRATÉGICO FINAL
# ----------------------------
st.divider()
st.header("🚀 Roadmap Estratégico de Execução")
rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown("### 📢 Aquisição de Mercado")
    st.markdown(f"""
    - **Ads de Persona:** Use imagens de **{get_mode('Occupation')}s** no trajeto diário.
    - **Foco Local:** Campanhas agressivas em **{get_mode('Region')}**.
    - **LinkedIn Ads:** Direcionar para graduados (**{get_mode('Education')}**).
    """)

with rec2:
    st.markdown("### 🔧 Operações e Produto")
    st.markdown(f"""
    - **B2B:** Leasing de frota para empresas com muitos **{get_mode('Occupation')}s**.
    - **Acessórios:** Kits para famílias com **{get_mode('Children')}** filhos.
    - **Serviço Premium:** Montagem em domicílio para **{get_mode('Home Owner')}s**.
    """)

with rec3:
    st.markdown("### 📈 Expansão e Escala")
    st.markdown(f"""
    - **Substituição de Carros:** Focar em lares com **{get_mode('Cars')}** carros.
    - **Eficiência Urbana:** Destacar durabilidade para trajetos de **{get_mode('Commute Distance')}**.
    - **Preço:** Ajustar condições para a faixa de **{get_mode('Faixa de Renda')}**.
    """)

st.success("""
✅ **Resumo Executivo Final:** A MozBikes não vende lazer, vende tempo e economia. O sucesso da próxima campanha depende de posicionar a bicicleta como um veículo de trabalho eficiente para profissionais urbanos com trajetos curtos.
""")
