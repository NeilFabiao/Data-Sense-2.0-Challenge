import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# 1. SETUP & IDENTIDADE VISUAL
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Analysis", layout="wide")

st.title("🚲 Dashboard Estratégico MozBikes")

# ----------------------------
# 2. CARREGAMENTO E TRATAMENTO
# ----------------------------
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

target = "Purchased Bike"

df["Income Group"] = pd.cut(
    df["Income"],
    bins=[0, 30000, 60000, 90000, 120000, float("inf")],
    labels=["<30k", "30k–60k", "60k–90k", "90k–120k", "120k+"]
)

buyers    = df[df[target] == "Yes"]
non_buyers = df[df[target] == "No"]

# ----------------------------
# DYNAMIC STAT CALCULATIONS
# ----------------------------
def get_mode(col_name):
    actual_col = next((c for c in buyers.columns if c.lower() == col_name.lower()), None)
    if actual_col and not buyers[actual_col].mode().empty:
        val = buyers[actual_col].mode()[0]
        if col_name.lower() == "home owner":
            return "Proprietário" if str(val).lower() == "yes" else "Inquilino"
        return str(val)
    return "N/A"

def pct_of_buyers(col_name, value):
    actual_col = next((c for c in buyers.columns if c.lower() == col_name.lower()), None)
    if actual_col is None: return "N/A"
    total = len(buyers)
    count = (buyers[actual_col] == value).sum()
    return f"{count / total * 100:.1f}%" if total > 0 else "N/A"

# Dynamic stats replacing hardcoded values
top_age_bracket  = get_mode("Age brackets")
top_occupation   = get_mode("Occupation")
top_commute      = get_mode("Commute Distance")
top_region       = get_mode("Region")
top_education    = get_mode("Education")

pct_age          = pct_of_buyers("Age brackets", top_age_bracket)
pct_occupation   = pct_of_buyers("Occupation", top_occupation)
pct_commute      = pct_of_buyers("Commute Distance", top_commute)
pct_region       = pct_of_buyers("Region", top_region)
pct_education    = pct_of_buyers("Education", top_education)

# Age range for dominant bracket
if "Age" in buyers.columns and "Age brackets" in buyers.columns:
    middle_age_buyers = buyers[buyers["Age brackets"] == top_age_bracket]["Age"]
    age_min = int(middle_age_buyers.min())
    age_max = int(middle_age_buyers.max())
    age_range_str = f"{age_min}–{age_max} anos"
else:
    age_range_str = top_age_bracket

# ----------------------------
# CONTEXT SECTION
# ----------------------------
st.markdown(f"""
### **📖 Contexto do Desafio: A Busca pelo Perfil Ideal**
Este projeto foi desenvolvido como um exercício estratégico para a **MozBikes**. O objetivo central foi identificar, através de variáveis demográficas e socioeconômicas, qual é o perfil de cliente com maior probabilidade de comprar uma bicicleta.

Como uma **tarefa oculta**, o desafio exigia que as conclusões fossem sustentadas por visualizações de dados claras, fornecendo recomendações com clareza e certeza para os stakeholders. 

#### **🎯 O Perfil "Golden": O Profissional Urbano de Meia-Idade**
Com base nos dados analisados, o perfil com maior probabilidade de conversão é composto por:
* **Idade:** Indivíduos de **{top_age_bracket}** (**{pct_age}** dos compradores) — faixa etária **{age_range_str}**.
* **Ocupação:** **{top_occupation}s** (**{pct_occupation}**).
* **Logística:** Pessoas que vivem a **{top_commute}** do trabalho (**{pct_commute}**).
* **Geografia:** Residentes da **{top_region}** (**{pct_region}**).
* **Educação:** Graduados com nível de **{top_education}** (**{pct_education}**).

**Justificativa Econômica e Comportamental:** Este grupo possui estabilidade financeira e alto poder aquisitivo. Comportamentalmente, a compra é motivada pela **eficiência**: para quem mora a menos de 1 milha do trabalho, a bicicleta é a solução de mobilidade mais lógica e rápida. Além disso, a quantidade de **carros** é o fator de maior influência na decisão, indicando que a bike é vista como um substituto estratégico ao veículo motorizado.
""")

# ----------------------------
# 3. SEÇÃO: APRENDIZADOS MOZ DEVS DATAWAVE 2.0
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
# 4. ANÁLISE VISUAL
# ----------------------------
st.divider()
st.header("📈 Visualização de Dados (Sustentação das Conclusões)")

def plot_bar_pie(feature, summary_text=""):
    if feature not in buyers.columns:
        st.warning(f"Coluna '{feature}' não encontrada nos dados.")
        return

    st.markdown(f"### **📊 Análise de {feature}**")

    # Sort by value count for nominal; by index for ordinal
    ordinal_features = ["Age brackets", "Commute Distance", "Income Group", "Children", "Cars"]
    if feature in ordinal_features:
        data = buyers[feature].value_counts().sort_index()
    else:
        data = buyers[feature].value_counts().sort_values(ascending=False)

    too_many_slices = len(data) > 5

    if too_many_slices:
        # Bar chart only for features with many categories
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(data.index.astype(str), data.values, color='#2E86C1', edgecolor='black')
        ax.set_title(f"Volume por {feature.upper()}", fontweight='bold')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{int(bar.get_height())}', ha='center', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(data.index.astype(str), data.values, color='#2E86C1', edgecolor='black')
            ax.set_title(f"Volume por {feature.upper()}", fontweight='bold')
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{int(bar.get_height())}', ha='center', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1',
                      '#FFC300', '#DAF7A6', '#C70039', '#900C3F', '#581845']
            ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140,
                   colors=colors[:len(data)], pctdistance=0.85)
            fig.gca().add_artist(plt.Circle((0, 0), 0.70, fc='white'))
            ax.set_title(f"Distribuição de {feature.upper()}", fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

    if summary_text:
        st.info(f"**Insight:** {summary_text}")

plot_bar_pie("Age brackets",         f"{top_age_bracket} domina {pct_age} das compras ({age_range_str}). Focar marketing neste público estável.")
plot_bar_pie("Commute Distance",     f"{pct_commute} moram a {top_commute} do trabalho. A bike é a solução logística ideal.")
plot_bar_pie("Occupation",           f"{top_occupation}s e técnicos são o motor de vendas ({pct_occupation}).")
plot_bar_pie("Cars",                 "Quem possui menos carros tem maior propensão a comprar para substituir o segundo veículo.")
plot_bar_pie("Income Group",         f"Público de renda {get_mode('Income Group')} é o mais lucrativo.")
plot_bar_pie("Marriedarital Status", "Os Solteiros tendem a dominar as compras — a motivação familiar e estabilidade financeira são fatores chave.")
plot_bar_pie("Gender",               "Distribuição por género revela segmentos distintos — campanhas podem ser personalizadas por género.")
plot_bar_pie("Children",             "Número de filhos influencia a necessidade de mobilidade alternativa e económica.")
plot_bar_pie("Education",            f"Nível {top_education} correlaciona com perfil de renda e propensão à compra consciente ({pct_education}).")
plot_bar_pie("Home Owner",           "Proprietários demonstram maior estabilidade financeira e maior conversão de compra.")
plot_bar_pie("Region",               f"{top_region} lidera as compras ({pct_region}). Geofencing regional pode maximizar o ROI de campanhas.")

# ----------------------------
# 5. ENGINE DE MACHINE LEARNING
# ----------------------------
st.divider()
st.header("🧠 Inteligência de Dados: O 'Golden Profile'")

tree_df = df.copy().dropna()
if "ID" in tree_df.columns:
    tree_df = tree_df.drop(columns=["ID"])

tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})

# Fix: separate encoder per column so inverse_transform works correctly
encoders = {}
categorical_cols = tree_df.select_dtypes(include="object").columns
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    tree_df[col] = encoders[col].fit_transform(tree_df[col])

X = tree_df.drop(columns=[target, "Income Group"])
y = tree_df[target]

# Train/test split for accuracy score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Refit on full data for feature importance
model.fit(X, y)
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance[importance > 0].sort_values()

col_ml1, col_ml2 = st.columns([1, 1.2])

with col_ml1:
    st.subheader("**🚀 Direcionadores de Decisão**")
    st.metric(label="Acurácia do Modelo (test set 20%)", value=f"{acc:.1%}")
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot(kind="barh", ax=ax, color='#117A65')
    for i, v in enumerate(importance):
        ax.text(v + 0.005, i, f'{v:.1%}', fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)

with col_ml2:
    st.subheader("**🧾📑📝 Perfil do Cliente Ideal**")
    full_persona = (
        f"O modelo identifica o comprador de alta probabilidade como um **{get_mode('Marriedarital Status')}** "
        f"do gênero **{get_mode('Gender')}** na faixa de **{top_age_bracket}** ({age_range_str}). "
        f"Geralmente um profissional de nível **{get_mode('Education')}** que trabalha como **{get_mode('Occupation')}**. "
        f"Ele é **{get_mode('Home Owner')}**, reside na região **{get_mode('Region')}**, tem **{get_mode('Children')} filho(s)** "
        f"e possui **{get_mode('Cars')} carro(s)**. Pertence à faixa de renda **{get_mode('Income Group')}**, "
        f"e o trajeto de **{get_mode('Commute Distance')}** é o perfeito para venda."
    )
    st.success(full_persona)

# ----------------------------
# 6. RECOMENDAÇÕES AUTOMÁTICAS (TODAS AS VARIÁVEIS, ORDENADAS POR IMPORTÂNCIA)
# ----------------------------
st.divider()
st.header("💰 Estratégia de Lucro (Gerada pelo Modelo)")
st.markdown(
    "Cada recomendação abaixo é **gerada automaticamente** com base no peso que o modelo de "
    "Machine Learning atribuiu a cada variável. Quanto maior a importância, maior a prioridade estratégica."
)

feature_recommendations = {
    "Cars": {
        "icon": "🚗",
        "title": "Substituição de Veículo",
        "action": (
            f"O número de carros é o **maior preditor de compra**. "
            f"Crie campanhas de economia mensal (combustível vs. bike) "
            f"direccionadas a donos de **{get_mode('Cars')} carro(s)**. "
            f"Mensagem chave: *'A sua bike paga-se sozinha em 3 meses.'*"
        )
    },
    "Income": {
        "icon": "💵",
        "title": "Segmentação por Renda",
        "action": (
            f"Renda é um driver de conversão forte. "
            f"Priorize a faixa **{get_mode('Income Group')}** com planos de financiamento e parcelamento. "
            f"Para rendas mais altas, posicione modelos premium com acessórios incluídos."
        )
    },
    "Age": {
        "icon": "🎯",
        "title": "Marketing por Faixa Etária",
        "action": (
            f"Idade influencia directamente a decisão de compra. "
            f"O segmento dominante é **{top_age_bracket}** ({age_range_str}). "
            f"Concentre os criativos nesta faixa com mensagens de saúde, produtividade e eficiência no dia-a-dia."
        )
    },
    "Age brackets": {
        "icon": "🎯",
        "title": "Marketing por Faixa Etária",
        "action": (
            f"Idade influencia directamente a decisão de compra. "
            f"O segmento dominante é **{top_age_bracket}** ({age_range_str}). "
            f"Concentre os criativos nesta faixa com mensagens de saúde, produtividade e eficiência no dia-a-dia."
        )
    },
    "Commute Distance": {
        "icon": "📍",
        "title": "Proximidade como Gatilho de Venda",
        "action": (
            f"A distância do trabalho é um gatilho decisivo de compra. "
            f"Lance campanhas geo-targeted para quem mora a **{get_mode('Commute Distance')}** do trabalho. "
            f"Parceria com apps de mapas (Google Maps, Waze) para interceptar este público no momento certo."
        )
    },
    "Occupation": {
        "icon": "💼",
        "title": "B2B e Parcerias Corporativas",
        "action": (
            f"Ocupação é um preditor relevante de conversão. "
            f"Feche acordos B2B com empresas que empregam **{get_mode('Occupation')}s**, "
            f"oferecendo planos corporativos com desconto por volume e manutenção incluída."
        )
    },
    "Marriedarital Status": {
        "icon": "👨‍👩‍👧",
        "title": "Apelo Familiar e Estabilidade",
        "action": (
            f"Estado civil influencia a propensão à compra. "
            f"Desenvolva campanhas para **{get_mode('Marriedarital Status')}s** "
            f"com foco em mobilidade familiar, segurança e economia doméstica partilhada."
        )
    },
    "Children": {
        "icon": "👶",
        "title": "Kits e Produtos Família",
        "action": (
            f"Número de filhos é um factor de decisão relevante. "
            f"Crie pacotes família com cadeirinhas, reboques e capacetes para quem tem "
            f"**{get_mode('Children')} filho(s)**. Promoções de back-to-school são ideais para este segmento."
        )
    },
    "Education": {
        "icon": "🎓",
        "title": "Canal Educado e Conteúdo Técnico",
        "action": (
            f"Escolaridade prediz conversão consciente e racional. "
            f"Use LinkedIn, podcasts e artigos técnicos para atingir graduados em **{get_mode('Education')}**. "
            f"Este público responde melhor a dados e comparações objectivas do que a apelos emocionais."
        )
    },
    "Region": {
        "icon": "🌍",
        "title": "Geofencing e Expansão Regional",
        "action": (
            f"Região é um factor de conversão significativo. "
            f"Concentre **80% do budget de marketing** em **{get_mode('Region')}** ({pct_region} dos compradores). "
            f"Para outras regiões, teste campanhas piloto antes de escalar o investimento."
        )
    },
    "Home Owner": {
        "icon": "🏠",
        "title": "Serviço Premium para Proprietários",
        "action": (
            f"Ser proprietário indica estabilidade financeira e maior ticket médio. "
            f"Ofereça serviços premium como montagem em domicílio, garantia estendida "
            f"e personalização de produto para **{get_mode('Home Owner')}s**."
        )
    },
    "Gender": {
        "icon": "⚡",
        "title": "Campanhas Personalizadas por Género",
        "action": (
            f"Género influencia preferências de produto e canal de comunicação. "
            f"Desenvolva criativos e linhas de produto distintos para o segmento "
            f"**{get_mode('Gender')}** dominante, sem negligenciar os restantes segmentos."
        )
    },
}

all_features_ranked = importance.sort_values(ascending=False)

for rank, (feature, score) in enumerate(all_features_ranked.items(), 1):
    rec = feature_recommendations.get(feature, None)

    if rec:
        icon   = rec["icon"]
        title  = rec["title"]
        action = rec["action"]
    else:
        icon   = "📊"
        title  = feature
        action = f"**{feature}** é um driver relevante com importância de {score:.1%}. Analise este segmento com prioridade."

    with st.expander(f"#{rank}  {icon}  {title}  —  Importância: {score:.1%}"):
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.metric(label="Peso no Modelo", value=f"{score:.1%}")
            st.markdown(f"**Variável:** `{feature}`")
        with col_b:
            st.markdown("**Acção Recomendada:**")
            st.info(action)

# ----------------------------
# 7. ROADMAP ESTRATÉGICO FINAL
# ----------------------------
st.divider()
st.header("🚀 Roadmap Estratégico MozBikes")

rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown("### 📢 Aquisição de Mercado")
    st.markdown(f"""
    - **Ads Persona:** Use imagens de **{get_mode('Occupation')}s** ({age_range_str}) no trajeto diário.
    - **Geofencing:** Focar 80% do budget na região de **{get_mode('Region')}** ({pct_region}).
    - **LinkedIn:** Canal prioritário para atingir graduados em **{get_mode('Education')}** ({pct_education}).
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
    - **Eficiência:** Otimizar durabilidade para trajetos de **{get_mode('Commute Distance')}** ({pct_commute}).
    - **Financiamento:** Facilitar para a faixa de renda **{get_mode('Income Group')}**.
    """)

st.success(f"""
✅ **Resumo Executivo Final:** O sucesso da MozBikes está na interseção da estabilidade profissional 
com o deslocamento urbano de curta distância. O modelo (acurácia: **{acc:.1%}**) recomenda focar 
em **{top_occupation}s** de **{top_age_bracket}** ({age_range_str}) na região de **{top_region}**, 
priorizando a **mobilidade profissional funcional** em vez de apenas lazer.
""")
