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
st.markdown(
    "Esta análise foca exclusivamente no perfil de clientes que **compraram** uma bicicleta, "
    "identificando padrões demográficos e comportamentais."
)

# ----------------------------
# LOAD DATA
# ----------------------------
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
except:
    st.error("Arquivo não encontrado. Verifique o nome do arquivo Excel.")
    st.stop()

df.columns = df.columns.str.strip()
target = "Purchased Bike"

# ----------------------------
# FILTER BUYERS
# ----------------------------
buyers = df[df[target] == "Yes"]

# ----------------------------
# FUNCTION: BAR + PIE
# ----------------------------
def plot_bar_pie(feature, summary_text=""):
    st.divider()
    st.subheader(f"📊 {feature} - Perfil dos Compradores")

    data = buyers[feature].value_counts()
    col1, col2 = st.columns(2)

    # BAR
    with col1:
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax, color='#1f77b4')
        ax.set_ylabel("Quantidade")
        ax.set_title(f"Distribuição por {feature}")
        st.pyplot(fig)

    # PIE
    with col2:
        fig, ax = plt.subplots()
        ax.pie(
            data,
            labels=data.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        )
        ax.set_title(f"Percentual por {feature}")
        ax.axis("equal")
        st.pyplot(fig)

    if summary_text:
        st.info(summary_text)

# ----------------------------
# ANALYSIS SECTIONS
# ----------------------------
if "Gender" in df.columns:
    plot_bar_pie(
        "Gender",
        "**Insight:** O mercado está equilibrado entre homens e mulheres."
    )

if "Education" in df.columns:
    plot_bar_pie(
        "Education",
        "**Insight:** A maioria dos compradores possui ensino superior."
    )

if "Occupation" in df.columns:
    plot_bar_pie(
        "Occupation",
        "**Insight:** Profissionais e trabalhadores qualificados dominam."
    )

if "Commute Distance" in df.columns:
    plot_bar_pie(
        "Commute Distance",
        "**Insight:** A maioria tem deslocamentos curtos."
    )

if "Age brackets" in df.columns:
    plot_bar_pie(
        "Age brackets",
        "**Insight:** A meia-idade domina as compras."
    )

# ----------------------------
# MACHINE LEARNING
# ----------------------------
st.divider()
st.title("🌳 What Drives Bike Purchases")

tree_df = df.copy().dropna()

# Remove ID if exists
if "ID" in tree_df.columns:
    tree_df = tree_df.drop(columns=["ID"])

# Encode target
tree_df[target] = tree_df[target].map({"Yes": 1, "No": 0})

# Encode categorical
le = LabelEncoder()
categorical_cols = tree_df.select_dtypes(include="object").columns

for col in categorical_cols:
    tree_df[col] = le.fit_transform(tree_df[col])

# Split
X = tree_df.drop(columns=[target])
y = tree_df[target]

# Train model
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X, y)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
st.subheader("🚲 Key Drivers of Bike Purchase")

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance[importance > 0].sort_values()

fig, ax = plt.subplots()
importance.plot(kind="barh", ax=ax, color='teal')
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# ----------------------------
# TOP FACTORS
# ----------------------------
st.subheader("🔥 Top Factors Explained")

top_factors = importance.sort_values(ascending=False).head(5)

if not top_factors.empty:
    for factor, score in top_factors.items():
        if factor == "Cars":
            st.write("**Cars:** Fewer cars → higher likelihood to buy a bike.")
        elif factor == "Commute Distance":
            st.write("**Commute Distance:** Short distances increase bike usage.")
        else:
            st.write(f"**{factor}:** Impact = {score:.2%}")
else:
    st.write("No significant drivers found.")
