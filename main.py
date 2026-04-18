import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. CONFIGURAÇÃO & TEMA
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Dashboard", layout="wide")

# --- INTRODUÇÃO E CONTEXTO DO DESAFIO ---
st.title("🚲 Dashboard Estratégico MozBikes")

st.markdown("""
### **📖 Contexto do Desafio: A Busca pelo Perfil Ideal**
Este projeto foi desenvolvido como um exercício estratégico para a **MozBikes**. O objetivo central foi identificar, através de variáveis demográficas e socioeconômicas, qual é o perfil de cliente com maior probabilidade de comprar uma bicicleta.

Como uma **tarefa oculta**, o desafio exigia que as conclusões fossem sustentadas por visualizações de dados claras, fornecendo **recomendações com clareza e certeza** para os stakeholders. 

#### **🎯 O Perfil "Golden": O Profissional Urbano de Meia-Idade**
Com base nos dados analisados, o perfil com maior probabilidade de conversão é composto por:
* **Idade:** Indivíduos de meia-idade (**79,6%** dos compradores).
* **Ocupação:** Profissionais (**31,2%**).
* **Logística:** Pessoas que vivem a **0-1 milha** do trabalho (**41,6%**).
* **Educação:** Graduados com nível de **Bacharelado** (**35,1%**).

**Justificativa Econômica e Comportamental:** Este grupo possui estabilidade financeira para investir em mobilidade de qualidade. Comportamentalmente, a compra é motivada pela **conveniência**: para quem mora a menos de 1 milha do trabalho, a bicicleta é a solução mais lógica, rápida e saudável.
""")

# ----------------------------
# 2. SEÇÃO: APRENDIZADOS MOZ DEVS DATAWAVE 2.0
# ----------------------------
st.divider()
st.header("🎓 Lições do Moz Devs DataWave 2.0")

dw_col1, dw_col2, dw_col3 = st.columns(3)

with dw_col1:
    st.markdown("#### **1. O Ecossistema e a Ferramenta**")
    st.info("""
    - **Caminho Profissional:** Engenheiro, Cientista e Analista de Dados ([Roadmap.sh](https://roadmap.sh)).
    - **A Analogia da Tilápia:** Conheça sua ferramenta. Às vezes não precisa de um bot, um **anzol simples** é o suficiente para o resultado esperado.
    - **Evolução:** Saímos da falta de dados para a abundância; o foco agora é a **mineração** estratégica.
    """)

with dw_col2:
    st.markdown("#### **2. Pilares da Maturidade**")
    st.warning("""
    2. **Qualidade de Dados:** A base da confiança.
    3. **Governança de Dados:** Regras e processos claros.
    4. **Soberania de Dados:** Controle estratégico.
    5. **Democratização:** Dados acessíveis para decisão.
    """)

with dw_col3:
    st.markdown("#### **3. Sua Jornada de Dados**")
    st.success("""
    - **Expectativa:** Insights básicos que respondam o problema.
    - **Simplicidade:** O simples que dá resposta é superior ao complexo que confunde.
    - **Inclusividade:** Design e labels feitos para quem "sabe zero" de dados.
    """)

# ----------------------------
# 3. CARREGAMENTO E VALIDAÇÃO
# ----------------------------
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

target = "Purchased Bike"
buyers = df[df[target] == "Yes"]

# ----------------------------
# 4. ANÁLISE VISUAL (O QUE OS DADOS SUSTENTAM)
# ----------------------------
st.divider()
st.header("📈 Visualização de Dados e Evidências")

def plot_bar_pie(feature, insight_text):
    if feature not in buyers.columns: return

    st.markdown(f"### **📊 Análise de {feature}**")
    data = buyers[feature].value_counts().sort_index()
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(data.index.astype(str), data.values, color='#3274A1', edgecolor='black')
        ax.set_title(f"Volume de Vendas por {feature}", fontweight='bold')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{int(bar.get_height())}', ha='center', fontweight='bold')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2']
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140, colors=colors, pctdistance=0.85)
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.set_title(f"Distribuição Percentual: {feature}", fontweight='bold')
        st.pyplot(fig)

    st.info(f"**Conclusão dos Dados:** {insight_text}")

# Chamadas baseadas no desafio
plot_bar_pie("Age brackets", "O público de meia-idade representa quase 80% das vendas. É um mercado de estabilidade, não de impulso.")
plot_bar_pie("Commute Distance", "41.6% dos compradores moram a menos de 1 milha. A proximidade é o maior preditor de compra.")
plot_bar_pie("Occupation", "Profissionais e Técnicos dominam. O produto é visto como ferramenta de produtividade urbana.")

# ----------------------------
# 5. RECOMENDAÇÕES FINAIS (O "PORQUÊ" E O "COMO")
# ----------------------------
st.divider()
st.header("🚀 Recomendações Estratégicas para Stakeholders")

# Estrutura de Resposta Curta e Direta conforme orientações
st.subheader("**O que foi feito e qual o resultado?**")
st.write("> Analisamos o dataset demográfico da MozBikes para identificar padrões de consumo. O resultado foi a descoberta de que a utilidade logística (distância curta) supera a renda como principal motivador de compra.")

st.markdown("### **💡 Recomendações com Clareza e Certeza**")

col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.info("#### **1. Estratégia de Marketing (Quem e Onde)**")
    st.markdown("""
    - **Foco Geográfico:** Concentrar investimentos de mídia em áreas com alta densidade de escritórios e residências num raio de 2km.
    - **Branding Profissional:** Criar campanhas que mostrem a bicicleta como o "transporte do sucesso" para o profissional de meia-idade, focando em tempo ganho e saúde.
    - **Segmentação:** Priorizar lares com **apenas 1 carro**, apresentando a bike como o substituto ideal para o segundo veículo.
    """)

with col_rec2:
    st.success("#### **2. Estratégia de Vendas (Como e Porquê)**")
    st.markdown("""
    - **Pitch de Venda Econômico:** Demonstrar o ROI (Retorno de Investimento) focado na economia de combustível e estacionamento para o trajeto diário curto.
    - **Parcerias B2B:** Oferecer planos de benefício corporativo para empresas que empregam profissionais e técnicos, facilitando a aquisição via folha de pagamento.
    - **Diferenciação:** Dado o nível de educação (Bachelors), utilizar argumentos técnicos e de impacto ambiental real na comunicação do produto.
    """)

st.divider()
st.warning("""
**Nota de Encerramento:** Seguindo os pilares do DataWave 2.0, este dashboard democratiza a informação ao transformar dados brutos em decisões de negócio simples, diretas e acionáveis.
""")
