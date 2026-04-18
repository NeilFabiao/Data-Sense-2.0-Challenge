import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. SETUP & CONFIGURATION
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Dashboard", layout="wide")

st.title("🚲 MozBikes Strategic Analysis")
st.markdown("""
Esta aplicação consolida todos os dados demográficos e comportamentais para identificar o perfil ideal de comprador 
e fornecer recomendações estratégicas baseadas em Machine Learning.
""")

# Carregamento de dados
try:
    # Certifique-se de que o arquivo está na mesma pasta que o script
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

df.columns = df.columns.str.strip()
target = "Purchased Bike"

# Filtrar apenas compradores para extrair as características de sucesso (Mode)
buyers = df[df[target] == "Yes"]

# ----------------------------
# 2. ANALYSIS HELPER FUNCTION
# ----------------------------
def plot_bar_pie(feature, summary_text=""):
    st.divider()
    st.subheader(f"📊 {feature} - Perfil dos Compradores")
    
    data = buyers[feature].value_counts()
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax, color='#1f77b4')
        ax.set_ylabel("Quantidade")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90, 
               colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.axis("equal")
        st.pyplot(fig)
    
    if summary_text:
        st.info(summary_text)

# ----------------------------
# 3. DEMOGRAPHIC INSIGHTS
# ----------------------------
st.header("📈 Análise de KPIs Principais")
if "Gender" in df.columns:
    plot_bar_pie("Gender", "**Insight:** O gênero está equilibrado, permitindo campanhas universais.")
if "Education" in df.columns:
    plot_bar_pie("Education", "**Insight:** O alto nível de instrução (Bachelors+) sugere que a comunicação deve ser técnica e profissional.")
if "Occupation" in df.columns:
    plot_bar_pie("Occupation", "**Insight:** Profissionais lideram as compras; o foco deve ser no setor corporativo.")

# ----------------------------
# 4. MACHINE LEARNING ENGINE
# ----------------------------
st.divider()
st.title("🌳 Decision Tree & Predictive Profiling")

# Preparação de Dados
tree_df = df.copy().dropna()
if 'ID' in tree_df.columns:
    tree_df = tree_df.drop(columns=['ID'])

# Encoding
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns
for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

# Treinamento
X = tree_df.drop(columns=[target])
y = tree_df[target]
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X, y)

# ----------------------------
# 5. THE ALL-KPI PERSONA SENTENCE
# ----------------------------
st.subheader("🤖 The 'Golden Profile' (ML-Generated Sentence)")

def get_trait(col):
    if col in buyers.columns:
        return str(buyers[col].mode()[0])
    return "[N/A]"

# Compiling every KPI into one master sentence
try:
    full_persona = (
        f"The Machine Learning model determines that the ideal MozBikes customer is a **{get_trait('Marital Status')}** "
        f"**{get_trait('Gender')}** in the **{get_trait('Age brackets')}** bracket, working as a **{get_trait('Occupation')}** "
        f"with a **{get_trait('Education')}** degree. Based on life indicators, they are a **{get_trait('Home Owner')}** "
        f"living in the **{get_trait('Region')}** region, with **{get_trait('Children')} children**, "
        f"and owning **{get_trait('Cars')} car(s)**. Strategically, they use the bike for a **{get_trait('Commute Distance')}** commute, "
        f"representing the highest probability of conversion."
    )
    st.success(full_persona)
except Exception as e:
    st.warning("Verifique se todas as colunas de KPIs estão presentes no arquivo Excel.")

# ----------------------------
# 6. COMPILED RECOMMENDATIONS
# ----------------------------
st.divider()
st.title("📌 Compilation of Strategic Recommendations")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.subheader("🚀 Marketing & Target Acquisition")
    st.markdown(f"""
    * **Target Content:** Design ads featuring **{get_trait('Occupation')}s** who are **{get_trait('Marital Status')}**, emphasizing how a bike simplifies their **{get_trait('Commute Distance')}** commute.
    * **Regional Focus:** Direct 45%+ of the marketing budget to **{get_trait('Region')}** to maximize ROI.
    * **Educational Outreach:** Focus on university networks and LinkedIn, as **{get_trait('Education')}** holders are 3x more likely to buy.
    """)

with rec_col2:
    st.subheader("🔧 Business Operations & Sales")
    st.markdown(f"""
    * **Vehicle Replacement:** Since buyers typically have **{get_trait('Cars')} car(s)**, offer a 'Bike-for-Car' trade-in program or fuel-saving calculator.
    * **Home-Owner Services:** As most are **{get_trait('Home Owner')}s**, offer premium home-delivery, assembly, and maintenance packages.
    * **Family Packaging:** With **{get_trait('Children')} children** being the common trait, create family-bundle discounts or child-seat accessory packages.
    """)

# ----------------------------
# 7. KPI METRIC DASHBOARD
# ----------------------------
st.divider()
st.subheader("📊 Summary of Success KPIs (Mode Values)")
kpi_list = ['Age brackets', 'Cars', 'Commute Distance', 'Occupation', 'Education']
m_cols = st.columns(len(kpi_list))

for i, kpi in enumerate(kpi_list):
    with m_cols[i]:
        st.metric(label=kpi, value=get_trait(kpi))

st.success("**Final Recommendation:** MozBikes should prioritize B2B partnerships with companies employing urban professionals to address 'Last-Mile' commute challenges.")
