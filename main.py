import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. SETUP
# ----------------------------
st.set_page_config(page_title="MozBikes Strategic Dashboard", layout="wide")

st.title("🚲 MozBikes Strategic Analysis")
st.markdown("Dashboard de inteligência comercial focado no perfil de conversão e recomendações de Machine Learning.")

# ----------------------------
# 2. LOAD DATA
# ----------------------------
try:
    df = pd.read_excel("Worked dataset- DataSense.xlsx", sheet_name="Working sheet")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

target = "Purchased Bike"

# ----------------------------
# 3. FILTER BUYERS
# ----------------------------
buyers = df[df[target] == "Yes"]

# ----------------------------
# 4. FUNCTION: BAR + PIE
# ----------------------------
def plot_bar_pie(feature, summary_text=""):
    if feature not in buyers.columns:
        return
        
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
            startangle=90
        )
        ax.set_title(f"Percentual por {feature}")
        ax.axis("equal")
        st.pyplot(fig)

    if summary_text:
        st.info(summary_text)

# ----------------------------
# 5. VISUAL ANALYSIS
# ----------------------------
st.header("📈 Buyer Profile Breakdown")

plot_bar_pie("Gender", "**Insight:** Distribuição equilibrada entre homens e mulheres.")
plot_bar_pie("Education", "**Insight:** Maioria com ensino superior.")
plot_bar_pie("Occupation", "**Insight:** Profissionais e técnicos dominam.")
plot_bar_pie("Age", "**Insight:** Idade ajuda a refinar o target com maior precisão.")
plot_bar_pie("Age brackets", "**Insight:** Meia-idade domina as compras.")
plot_bar_pie("Commute Distance", "**Insight:** Forte presença de trajetos curtos.")
plot_bar_pie("Home Owner", "**Insight:** Indica estabilidade financeira do cliente.")

# ----------------------------
# 6. MACHINE LEARNING
# ----------------------------
st.divider()
st.header("🌳 What Drives Bike Purchases")

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
# 7. FEATURE IMPORTANCE
# ----------------------------
st.subheader("🚲 Key Drivers of Bike Purchase")

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance[importance > 0].sort_values()

fig, ax = plt.subplots()
importance.plot(kind="barh", ax=ax, color='teal')
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# ----------------------------
# 8. TOP 3 DRIVERS
# ----------------------------
st.subheader("🔥 Top 3 Drivers")

top3 = importance.sort_values(ascending=False).head(3)

for i, (factor, score) in enumerate(top3.items(), 1):
    st.write(f"{i}️⃣ **{factor}** — Impact: {score:.2%}")

# ----------------------------
# 9. PERSONA (FIXED)
# ----------------------------
st.divider()
st.header("🧠 Ideal Customer Profile")

persona_df = df.copy().dropna()

if "ID" in persona_df.columns:
    persona_df = persona_df.drop(columns=["ID"])

persona_df = persona_df[persona_df[target] == "Yes"]

def get_mode(col):
    if col in persona_df.columns and not persona_df[col].mode().empty:
        return persona_df[col].mode()[0]
    return "N/A"

persona = {
    "Gender": get_mode("Gender"),
    "Age": get_mode("Age"),
    "AgeBracket": get_mode("Age brackets"),
    "Occupation": get_mode("Occupation"),
    "Education": get_mode("Education"),
    "Region": get_mode("Region"),
    "Children": get_mode("Children"),
    "Cars": get_mode("Cars"),
    "Commute": get_mode("Commute Distance"),
    "HomeOwner": get_mode("Home Owner")
}

# Handle Yes/No or 1/0
home_status = "homeowner" if persona["HomeOwner"] in ["Yes", 1] else "non-homeowner"

st.markdown(f"""
Based on the Decision Tree analysis and buyer distribution, the **highest-probability MozBikes customer** is a **{persona['AgeBracket']} {persona['Gender']} professional**, approximately **{persona['Age']} years old**.

This individual typically:
- Works in a **{persona['Occupation']} role**
- Holds a **{persona['Education']} qualification**
- Lives in **{persona['Region']}**
- Is a **{home_status}**
- Has **{persona['Children']} children** and owns **{persona['Cars']} car(s)**

From a behavioral standpoint, their **{persona['Commute']} commute** indicates a strong preference for **short-distance, efficient transportation**.

👉 This profile represents a **financially stable, urban working professional**, making them the ideal target for MozBikes' mobility solutions.
""")

# ----------------------------
# 10. STRATEGIC ACTIONS
# ----------------------------
st.divider()
st.header("🚀 Key Strategic Actions")

st.markdown("""
### 1️⃣ Vehicle Ownership (Cars)
- Fewer cars → higher likelihood to buy  
- Position bikes as a **car alternative**

### 2️⃣ Commute Distance
- Short trips dominate  
- Focus on **urban mobility & last-mile transport**

### 3️⃣ Occupation
- Professionals dominate buyers  
- Target workplaces & corporate programs  

---

💡 **Strategic Insight:**  
MozBikes is solving a **daily transport problem**, not selling leisure products.
""")

# ----------------------------
# 11. FINAL TAKEAWAY
# ----------------------------
st.success("""
✅ The ideal MozBikes customer is a working professional with a short commute, using bicycles as a practical alternative to cars.
""")
