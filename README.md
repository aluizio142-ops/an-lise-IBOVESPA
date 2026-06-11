# 📈 Previsão de Tendência e Magnitude do IBOVESPA com Machine Learning e Deep Learning

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Data Science](https://img.shields.io/badge/Área-Finanças%20%2F%20Análise%20de%20Risco-green)
![Machine Learning](https://img.shields.io/badge/Modelos-MLP%20%7C%20SARIMAX%20%7C%20Ensembles-orange)

## 🎯 Contexto do Problema e Objetivo de Negócio
Prever a movimentação de índices acionários é um dos maiores desafios do mercado financeiro devido à alta volatilidade, ruídos e à natureza não-linear dos ativos de renda variável. Para bancos, fintechs e mesas de operações, antecipar com precisão o comportamento do principal índice da bolsa brasileira (**IBOVESPA**) é um diferencial estratégico indispensável para a mitigação de riscos, alocação inteligente de capital e estratégias de *hedging*.

O objetivo deste projeto é construir um pipeline completo de Data Science para **prever tanto a magnitude de preço quanto a acurácia direcional (tendência de alta ou baixa) do fechamento diário do IBOVESPA**, utilizando uma base de dados histórica robusta que abrange três décadas de mercado (1995 a 2025).

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Linguagem:** Python 3.12+
* **Manipulação e Análise Temporal:** `Pandas`, `NumPy`
* **Análise Quantitativa e Indicadores Técnicos:** `pandas-ta` (Python Technical Analysis)
* **Modelagem Preditiva & Machine Learning:** `Scikit-Learn` e `Statsmodels`
* **Modelos Avaliados no Benchmarking:**
  * *Modelos Lineares & Regressão:* Linear Regression, Lasso, ElasticNet, SVR.
  * *Baseados em Árvores & Ensembles:* Decision Tree (CART), K-Neighbors, AdaBoost, Gradient Boosting, RandomForest e ExtraTrees.
  * *Redes Neurais Preditivas:* Multi-Layer Perceptron (MLP).
  * *Séries Temporais Clássicas:* SARIMAX.

---

## 🧬 Engenharia de Recursos (Feature Engineering)
Para capturar os momentos de sobrecompra, sobrevenda, volatilidade e momentum do mercado, o pipeline transforma os dados brutos de preço (`open`, `close`, `high`, `low`) utilizando indicadores técnicos consagrados pelo mercado financeiro:

1. **Indicadores de Momentum & Tendência:**
   * **RSI (Relative Strength Index):** Captura a força dos movimentos de preço e exaustão de tendência.
   * **MACD (Moving Average Convergence Divergence):** Identifica mudanças na força, direção e momentum do ativo.
   * **AO (Awesome Oscillator):** Oscilador fantástico para medir o momentum do mercado.
2. **Indicadores de Volatilidade:**
   * **Bollinger Bands (%B):** Mensura a dispersão do preço em relação à média móvel.
3. **Métricas de Distância e Defasagem (Lags):**
   * Cálculo da distância percentual do preço de fechamento em relação à **EMA de 9 períodos** (`distance_to_ema9_d-1`).
   * Retornos defasados de 10 dias (`ROC_10` e `MOM_10`) para avaliar a persistência da tendência.

---

## 🔬 Estratégia de Validação e Benchmarking Iterativo

O projeto foi conduzido através de 3 ciclos analíticos incrementais para garantir o máximo desempenho e evitar o *Data Leakage* (vazamento de dados temporais):

### Fase 1: Benchmarking Inicial (Modelos Padrão)
* Separação estrita dos dados respeitando a cronologia de Séries Temporais (Treino histórico e os últimos 30 dias para Teste cego).
* Avaliação preliminar de 11 algoritmos utilizando o erro de magnitude. Modelos Lineares apresentaram melhor capacidade inicial de generalização, enquanto modelos de árvore sofreram forte *overfitting* nas configurações padrão.

### Fase 2: Enriquecimento com Indicadores Técnicos & Otimização Temporal
* Inclusão dos indicadores gerados pelo `pandas-ta`.
* Modelagem especializada do **SARIMAX (2,0,0)** com regressores exógenos e processamento de redes neurais.
* **Resultado:** Redução expressiva do MAE e MAPE geral, comprovando que os indicadores técnicos adicionaram forte poder preditivo ao modelo.

### Fase 3: Escalonamento Avançado & Tuning de Hiperparâmetros
* Aplicação do `StandardScaler` para padronizar a escala de indicadores com naturezas distintas.
* **Ajuste Fino (Grid Search CV com 10 Folds):** Otimização dos parâmetros da Rede Neural MLP (`activation`, `alpha`, `hidden_layer_sizes`, `learning_rate_init`, `solver`).

---

## 📊 Resultados e Métricas de Negócio (Conjunto de Teste)

Os modelos foram desafiados em um cenário cego de 30 dias de mercado. Dependendo do objetivo do negócio (Operação Quantitativa ou Gestão de Risco), dois modelos se consagraram vencedores:

### 1. Campeão em Previsão de Valor (Magnitude)
Os modelos lineares (**Linear Regression** e **Lasso**) obtiveram a maior precisão na aproximação do valor real do índice:
* **MAPE (Erro Percentual Absoluto Médio):** **0.53%** no conjunto de teste.
* **Acurácia Direcional:** **60.00%** de acerto na tendência do dia.

### 2. Campeão em Previsão de Tendência (Tomada de Decisão Quantitativa)
Após o escalonamento de features e o refinamento por Grid Search, a Rede Neural **MLP (Multi-Layer Perceptron)** se consolidou como a ferramenta ideal para estratégias de tendência:
* **Acurácia Direcional (Direção Correta do Fechamento):** **76.67%** (Acerto de 23 dos 30 dias de teste).
* **Maturidade de Risco:** Superou a meta de corte de 75%, tornando-se um algoritmo altamente confiável para tomada de decisão buy/sell.

---

## 🚀 Como Executar o Projeto

1. Clone o repositório:
```bash
git clone [https://github.com/aluizio142-ops/an-lise-IBOVESPA.git](https://github.com/aluizio142-ops/an-lise-IBOVESPA.git)
