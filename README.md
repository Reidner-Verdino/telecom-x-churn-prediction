# 📡 Telecom X — Previsão de Evasão de Clientes (Churn Prediction)

> **Desafio de Machine Learning** | Parte 2 — Pipeline de Modelagem Preditiva

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completo-green)

---

## 🎯 Objetivo

Desenvolver modelos preditivos capazes de identificar clientes com maior probabilidade de cancelar seus serviços, permitindo à Telecom X agir proativamente com estratégias de retenção.

---

## 📁 Estrutura do Repositório

```
telecom-churn/
│
├── data/
│   └── telecom_tratado.csv          # Dataset tratado (Parte 1)
│
├── notebooks/
│   └── TelecomX_Churn_Prediction.ipynb  # Notebook principal
│
├── outputs/
│   ├── 01_proporcao_evasao.png
│   ├── 02_correlacao_evasao.png
│   ├── 03_heatmap_correlacao.png
│   ├── 04_analises_direcionadas.png
│   ├── 05_comparativo_metricas.png
│   ├── 06_matrizes_confusao.png
│   ├── 07_curvas_roc.png
│   ├── 08_importancia_rf.png
│   └── 09_coeficientes_lr.png
│
├── reports/
│   └── relatorio_telecom_churn.html  # Relatório completo
│
├── requirements.txt
└── README.md
```

---

## 🔧 Pipeline de Machine Learning

| Etapa | Descrição |
|---|---|
| 1 | Carregamento dos dados tratados |
| 2 | Remoção de colunas irrelevantes (IDs) |
| 3 | One-Hot Encoding das variáveis categóricas |
| 4 | Análise de proporção de evasão e balanceamento |
| 5 | Normalização (StandardScaler — Regressão Logística) |
| 6 | Análise de correlação e heatmap |
| 7 | Análises direcionadas (contrato × evasão, etc.) |
| 8 | Treinamento de 3 modelos de classificação |
| 9 | Avaliação com métricas (Acurácia, F1, AUC-ROC) |
| 10 | Importância das variáveis + Conclusão estratégica |

---

## 🤖 Modelos Treinados

| Modelo | Normalização | Justificativa |
|---|---|---|
| **Regressão Logística** | ✅ Sim (StandardScaler) | Interpretável, sensível à escala, boa para baseline |
| **Random Forest** | ❌ Não necessário | Ensemble robusto, importância nativa de features |
| **Árvore de Decisão** | ❌ Não necessário | Alta interpretabilidade, boa referência comparativa |

---

## 📊 Resultados

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| ⭐ Regressão Logística | 67,78% | 49,30% | **71,24%** | **58,27%** | **76,18%** |
| 🌲 Random Forest | 70,69% | **53,20%** | 59,78% | 56,30% | 76,15% |
| 🌿 Árvore de Decisão | 66,50% | 47,92% | 69,89% | 56,86% | 71,35% |

> **Modelo recomendado:** Regressão Logística (melhor Recall e F1-Score — ideal para detectar o máximo de clientes em risco de evasão)

---

## 🔍 Principais Fatores de Evasão

1. 📅 **Contrato Mês a Mês** — Principal preditor (taxa ~3x maior)
2. ⏱ **Tempo de Contrato Curto** (< 12 meses) — Alta vulnerabilidade
3. 💰 **Cobrança Mensal Alta** — Sensibilidade ao preço
4. 🌐 **Internet Fibra Óptica** — Possível insatisfação com custo/qualidade
5. 💳 **Pagamento via Cheque Eletrônico** — Menor fidelização

---

## 🚀 Estratégias de Retenção

- **Migração para contratos anuais** com incentivos e descontos
- **Programa de fidelidade** para clientes com < 12 meses
- **Revisão do plano de fibra** (preço, qualidade, suporte)
- **Incentivo ao débito automático** (desconto de 5%)
- **Scoring de churn em tempo real** para ações proativas

---

## 🛠️ Como Executar

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/telecom-churn.git
cd telecom-churn

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar o notebook
jupyter notebook notebooks/TelecomX_Churn_Prediction.ipynb
```

---

## 📦 Dependências

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
imbalanced-learn>=0.10  # opcional (SMOTE)
```

---

## 👤 Autor

Desenvolvido como parte do **Desafio Telecom X** — Programa de Data Science

---

*Pipeline: Pré-processamento → Encoding → Normalização → Modelagem → Avaliação → Insights Estratégicos*
