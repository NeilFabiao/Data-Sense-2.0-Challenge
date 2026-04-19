# 🚲 MozBikes Strategic Analysis Dashboard

Este projeto foi desenvolvido como parte do desafio **Moz Devs DataWave 2.0**. A aplicação é um Dashboard Estratégico que utiliza Ciência de Dados e Machine Learning para identificar o "Perfil Golden" de clientes da **MozBikes**, transformando dados brutos em recomendações acionáveis para marketing e vendas.

## 🎯 O Desafio: A Busca pelo Perfil Ideal

O objetivo central foi responder à pergunta: **"Qual é o perfil de cliente com maior probabilidade de comprar uma bicicleta?"**

Para isso, o projeto analisa variáveis demográficas e socioeconómicas, sustentando conclusões através de modelos preditivos e visualizações claras, permitindo que os stakeholders tomem decisões baseadas em dados (Data-Driven).

## 🚀 Funcionalidades Principais

* **Identificação do "Perfil Golden":** Cálculo dinâmico das características dominantes (idade, ocupação, região e escolaridade) do comprador ideal.
* **Análise Visual Sustentada:** Gráficos interativos (Barras e Pie Charts) que explicam o comportamento de cada segmento.
* **Engine de Machine Learning:** * Uso de **Árvore de Decisão** para prever a propensão de compra.
    * Cálculo de **Importância das Variáveis** (Feature Importance) para saber o que realmente move o ponteiro das vendas.
* **Estratégia Automática:** Geração de recomendações de marketing baseadas nos pesos atribuídos pelo modelo de ML (ex: foco em substituição de veículos, geofencing ou parcerias B2B).
* **Roadmap Estratégico:** Plano de ação dividido em Aquisição, Operações e Escala.

## 🛠️ Tecnologias Utilizadas

* **Python**: Linguagem principal.
* **Streamlit**: Interface web e dashboard.
* **Pandas & Openpyxl**: Manipulação e leitura de dados Excel.
* **Matplotlib**: Visualizações gráficas.
* **Scikit-Learn**: Modelagem preditiva (DecisionTreeClassifier) e pré-processamento (LabelEncoder).

## 📦 Como Instalar e Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/mozbikes-dashboard.git](https://github.com/seu-usuario/mozbikes-dashboard.git)
    cd mozbikes-dashboard
    ```

2.  **Instale as dependências:**
    ```bash
    pip install streamlit pandas matplotlib scikit-learn openpyxl
    ```

3.  **Prepare o Dataset:**
    Certifique-se de que o ficheiro `Worked dataset- DataSense.xlsx` está na raiz do projeto.

4.  **Execute a aplicação:**
    ```bash
    streamlit run seu_arquivo.py
    ```

## 🎓 Lições do Moz Devs DataWave 2.0

O desenvolvimento deste dashboard seguiu os pilares de maturidade de dados discutidos no evento:
1.  **Qualidade de Dados:** Limpeza e tratamento de strings e categorias.
2.  **Maturidade Analítica:** Evolução da análise descritiva para a preditiva.
3.  **Democratização:** Transformar logs e tabelas complexas em insights visuais compreensíveis para qualquer gestor.
4.  **Soberania e Ação:** Não apenas mostrar o "o quê", mas recomendar o "como" agir.

---
**Desenvolvido como parte do ecossistema Moz Devs.**
*"O simples que dá resposta ganha do complexo que confunde."*
