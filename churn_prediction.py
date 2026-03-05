"""
╔══════════════════════════════════════════════════════════════════╗
║         TELECOM X — PREVISÃO DE EVASÃO DE CLIENTES (CHURN)      ║
║                        Parte 2 — Modelagem                       ║
╚══════════════════════════════════════════════════════════════════╝

Objetivos:
  1. Pré-processamento e encoding
  2. Análise de correlação e seleção de variáveis
  3. Treinamento de modelos (Regressão Logística + Random Forest)
  4. Avaliação com métricas (Acurácia, Precisão, Recall, F1, Matriz)
  5. Importância das variáveis
  6. Conclusão estratégica
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# ─── Configurações visuais ─────────────────────────────────────────────────────
PALETA = {
    'primario':  '#1A1F36',
    'acento':    '#E94560',
    'positivo':  '#2ECC71',
    'neutro':    '#F0F4FF',
    'texto':     '#1A1F36',
    'cinza':     '#7F8C8D',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFBFF',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

OUTPUT = '/mnt/user-data/outputs/'
import os; os.makedirs(OUTPUT, exist_ok=True)

print("=" * 65)
print("  TELECOM X — PREVISÃO DE EVASÃO DE CLIENTES")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# ETAPA 1 — CARREGAMENTO E INSPEÇÃO DOS DADOS
# ══════════════════════════════════════════════════════════════════
print("\n[1/10] Carregando dados tratados...")
df = pd.read_csv('/home/claude/telecom_churn/data/telecom_tratado.csv')
print(f"  → Shape: {df.shape}")
print(f"  → Colunas: {list(df.columns)}")
print(f"  → Valores nulos: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════
# ETAPA 2 — REMOÇÃO DE COLUNAS IRRELEVANTES
# ══════════════════════════════════════════════════════════════════
print("\n[2/10] Verificando colunas irrelevantes...")
# Neste dataset já tratado, não há ID. Mas registramos a lógica.
colunas_iniciais = df.shape[1]
# Se houvesse 'id_cliente', faríamos: df = df.drop(columns=['id_cliente'])
print(f"  → Colunas mantidas: {df.shape[1]} (nenhuma coluna de ID encontrada)")

# ══════════════════════════════════════════════════════════════════
# ETAPA 3 — ENCODING DE VARIÁVEIS CATEGÓRICAS
# ══════════════════════════════════════════════════════════════════
print("\n[3/10] Aplicando One-Hot Encoding...")

categoricas = df.select_dtypes(include='object').columns.tolist()
print(f"  → Variáveis categóricas: {categoricas}")

df_encoded = pd.get_dummies(df, columns=categoricas, drop_first=False)

# Converter booleanos gerados pelo get_dummies para int
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print(f"  → Shape após encoding: {df_encoded.shape}")

# ══════════════════════════════════════════════════════════════════
# ETAPA 4 — PROPORÇÃO DE EVASÃO
# ══════════════════════════════════════════════════════════════════
print("\n[4/10] Analisando proporção de evasão...")
contagem = df['evasao'].value_counts()
proporcao = df['evasao'].value_counts(normalize=True) * 100
print(f"  → Clientes ATIVOS:  {contagem[0]:,} ({proporcao[0]:.1f}%)")
print(f"  → Clientes EVADIDOS: {contagem[1]:,} ({proporcao[1]:.1f}%)")
print(f"  → Razão de desequilíbrio: 1:{contagem[0]/contagem[1]:.1f}")

# Gráfico — Proporção de Evasão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distribuição da Evasão de Clientes', fontsize=16,
             fontweight='bold', color=PALETA['primario'], y=1.02)

cores = [PALETA['positivo'], PALETA['acento']]
labels = ['Ativo', 'Evadido']
wedge_props = {'edgecolor': 'white', 'linewidth': 3}

# Pizza
axes[0].pie(contagem.values, labels=labels, colors=cores, autopct='%1.1f%%',
            startangle=90, wedgeprops=wedge_props, textprops={'fontsize': 13})
axes[0].set_title('Proporção de Evasão', fontsize=13, fontweight='bold',
                  color=PALETA['primario'])

# Barras
bars = axes[1].bar(labels, contagem.values, color=cores, edgecolor='white',
                   linewidth=1.5, width=0.5)
for bar, val in zip(bars, contagem.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}', ha='center', va='bottom', fontweight='bold',
                 fontsize=13, color=PALETA['primario'])
axes[1].set_title('Contagem por Classe', fontsize=13, fontweight='bold',
                  color=PALETA['primario'])
axes[1].set_ylabel('Número de Clientes', fontsize=11)
axes[1].set_facecolor('#FAFBFF')

plt.tight_layout()
plt.savefig(OUTPUT + '01_proporcao_evasao.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Gráfico salvo: 01_proporcao_evasao.png")

# ══════════════════════════════════════════════════════════════════
# ETAPA 5 — NORMALIZAÇÃO (apenas para Regressão Logística)
# ══════════════════════════════════════════════════════════════════
print("\n[5/10] Preparando dados para modelagem...")

TARGET = 'evasao'
X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

feature_names = X.columns.tolist()

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  → Treino: {X_train.shape[0]:,} amostras")
print(f"  → Teste:  {X_test.shape[0]:,} amostras")

# Normalização para modelos sensíveis à escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("  → StandardScaler aplicado para Regressão Logística")

# ══════════════════════════════════════════════════════════════════
# ETAPA 6 — ANÁLISE DE CORRELAÇÃO
# ══════════════════════════════════════════════════════════════════
print("\n[6/10] Gerando análise de correlação...")

# Correlação com a variável alvo
corr_target = df_encoded.corr(numeric_only=True)['evasao'].drop('evasao').sort_values(key=abs, ascending=False)
top15 = corr_target.head(15)

# Gráfico de correlação com alvo
fig, ax = plt.subplots(figsize=(11, 7))
cores_barras = [PALETA['acento'] if v > 0 else PALETA['positivo'] for v in top15.values]
bars = ax.barh(top15.index[::-1], top15.values[::-1], color=cores_barras[::-1],
               edgecolor='white', linewidth=1)
for bar, val in zip(bars, top15.values[::-1]):
    ax.text(val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=9, fontweight='bold', color=PALETA['primario'])

ax.axvline(0, color=PALETA['cinza'], linewidth=1, linestyle='--')
ax.set_title('Top 15 Variáveis — Correlação com Evasão', fontsize=14,
             fontweight='bold', color=PALETA['primario'], pad=15)
ax.set_xlabel('Correlação de Pearson', fontsize=11)
patch_pos = mpatches.Patch(color=PALETA['acento'], label='Correlação positiva (aumenta evasão)')
patch_neg = mpatches.Patch(color=PALETA['positivo'], label='Correlação negativa (reduz evasão)')
ax.legend(handles=[patch_pos, patch_neg], loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT + '02_correlacao_evasao.png', dpi=150, bbox_inches='tight')
plt.close()

# Heatmap das variáveis numéricas
num_cols = ['tempo_contrato_meses', 'cobranca_mensal', 'total_gasto', 'evasao', 'idoso']
corr_matrix = df_encoded[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, ax=ax, mask=mask,
            linewidths=2, linecolor='white',
            annot_kws={'size': 11, 'weight': 'bold'})
ax.set_title('Matriz de Correlação — Variáveis Numéricas', fontsize=13,
             fontweight='bold', color=PALETA['primario'], pad=15)
plt.tight_layout()
plt.savefig(OUTPUT + '03_heatmap_correlacao.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Gráficos de correlação salvos")

# ══════════════════════════════════════════════════════════════════
# ETAPA 7 — ANÁLISES DIRECIONADAS
# ══════════════════════════════════════════════════════════════════
print("\n[7/10] Análises direcionadas (Contrato × Evasão, Gasto × Evasão)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análises Direcionadas — Fatores de Evasão', fontsize=15,
             fontweight='bold', color=PALETA['primario'], y=1.01)

# 1) Boxplot: Tempo de contrato × Evasão
labels_ev = ['Ativo', 'Evadido']
data_tempo = [df[df['evasao']==0]['tempo_contrato_meses'],
              df[df['evasao']==1]['tempo_contrato_meses']]
bp = axes[0,0].boxplot(data_tempo, labels=labels_ev, patch_artist=True,
                        medianprops={'color': 'white', 'linewidth': 2})
for patch, color in zip(bp['boxes'], [PALETA['positivo'], PALETA['acento']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0,0].set_title('Tempo de Contrato × Evasão', fontsize=12, fontweight='bold')
axes[0,0].set_ylabel('Meses de Contrato', fontsize=10)

# 2) Boxplot: Cobrança mensal × Evasão
data_cob = [df[df['evasao']==0]['cobranca_mensal'],
            df[df['evasao']==1]['cobranca_mensal']]
bp2 = axes[0,1].boxplot(data_cob, labels=labels_ev, patch_artist=True,
                         medianprops={'color': 'white', 'linewidth': 2})
for patch, color in zip(bp2['boxes'], [PALETA['positivo'], PALETA['acento']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0,1].set_title('Cobrança Mensal × Evasão', fontsize=12, fontweight='bold')
axes[0,1].set_ylabel('R$ / mês', fontsize=10)

# 3) Tipo de contrato × Evasão
contrato_churn = df.groupby('tipo_contrato')['evasao'].mean() * 100
bars3 = axes[1,0].bar(contrato_churn.index, contrato_churn.values,
                       color=[PALETA['acento'], PALETA['positivo'], '#3498DB'],
                       edgecolor='white', linewidth=1.5)
for bar, val in zip(bars3, contrato_churn.values):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
axes[1,0].set_title('Taxa de Evasão por Tipo de Contrato', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('Taxa de Evasão (%)', fontsize=10)
axes[1,0].tick_params(axis='x', labelsize=9)

# 4) Internet × Evasão
internet_churn = df.groupby('servico_internet')['evasao'].mean() * 100
bars4 = axes[1,1].bar(internet_churn.index, internet_churn.values,
                       color=[PALETA['acento'], '#3498DB', PALETA['positivo']],
                       edgecolor='white', linewidth=1.5)
for bar, val in zip(bars4, internet_churn.values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
axes[1,1].set_title('Taxa de Evasão por Tipo de Internet', fontsize=12, fontweight='bold')
axes[1,1].set_ylabel('Taxa de Evasão (%)', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT + '04_analises_direcionadas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Gráficos de análise salvos")

# ══════════════════════════════════════════════════════════════════
# ETAPA 8 — CRIAÇÃO E TREINAMENTO DOS MODELOS
# ══════════════════════════════════════════════════════════════════
print("\n[8/10] Treinando modelos...")

# Modelo 1: Regressão Logística (com normalização)
print("  → Treinando Regressão Logística...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Modelo 2: Random Forest (sem normalização)
print("  → Treinando Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                             min_samples_split=10, random_state=42,
                             class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Modelo 3: Árvore de Decisão (bônus)
print("  → Treinando Árvore de Decisão...")
dt = DecisionTreeClassifier(max_depth=8, min_samples_split=20,
                             random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

# ══════════════════════════════════════════════════════════════════
# ETAPA 9 — AVALIAÇÃO DOS MODELOS
# ══════════════════════════════════════════════════════════════════
print("\n[9/10] Avaliando modelos...")

def calcular_metricas(nome, y_true, y_pred, y_prob):
    return {
        'Modelo': nome,
        'Acurácia':   round(accuracy_score(y_true, y_pred)  * 100, 2),
        'Precisão':   round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        'Recall':     round(recall_score(y_true, y_pred)    * 100, 2),
        'F1-Score':   round(f1_score(y_true, y_pred)        * 100, 2),
        'AUC-ROC':    round(roc_auc_score(y_true, y_prob)   * 100, 2),
    }

resultados = [
    calcular_metricas('Regressão Logística', y_test, y_pred_lr, y_prob_lr),
    calcular_metricas('Random Forest',       y_test, y_pred_rf, y_prob_rf),
    calcular_metricas('Árvore de Decisão',   y_test, y_pred_dt, y_prob_dt),
]
df_resultados = pd.DataFrame(resultados)
print("\n  COMPARATIVO DE MÉTRICAS:")
print(df_resultados.to_string(index=False))

# ─── Gráfico de Métricas Comparativas ────────────────────────────
metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
modelos   = df_resultados['Modelo'].tolist()
cores_mod = [PALETA['acento'], PALETA['primario'], '#3498DB']
x = np.arange(len(metricas))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
for i, (modelo, cor) in enumerate(zip(modelos, cores_mod)):
    vals = df_resultados[df_resultados['Modelo'] == modelo][metricas].values[0]
    bars = ax.bar(x + i*width, vals, width, label=modelo, color=cor,
                  alpha=0.88, edgecolor='white', linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', color=PALETA['primario'])

ax.set_xticks(x + width)
ax.set_xticklabels(metricas, fontsize=11)
ax.set_ylim(0, 110)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Comparativo de Desempenho dos Modelos', fontsize=14,
             fontweight='bold', color=PALETA['primario'], pad=15)
ax.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(OUTPUT + '05_comparativo_metricas.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Matrizes de Confusão ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Matrizes de Confusão', fontsize=14, fontweight='bold',
             color=PALETA['primario'])

for ax, y_pred, nome, cor in zip(axes,
    [y_pred_lr, y_pred_rf, y_pred_dt],
    modelos, cores_mod):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(cor, as_cmap=True),
                linewidths=2, linecolor='white',
                xticklabels=['Ativo', 'Evadido'],
                yticklabels=['Ativo', 'Evadido'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(nome, fontsize=11, fontweight='bold', color=PALETA['primario'])
    ax.set_ylabel('Real', fontsize=10)
    ax.set_xlabel('Previsto', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT + '06_matrizes_confusao.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Curvas ROC ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for y_prob, nome, cor in zip([y_prob_lr, y_prob_rf, y_prob_dt], modelos, cores_mod):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=cor, lw=2.5, label=f'{nome} (AUC = {auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Classificador Aleatório')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
ax.set_title('Curvas ROC — Comparativo de Modelos', fontsize=14,
             fontweight='bold', color=PALETA['primario'])
ax.legend(fontsize=11, loc='lower right')
plt.tight_layout()
plt.savefig(OUTPUT + '07_curvas_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Gráficos de avaliação salvos")

# ══════════════════════════════════════════════════════════════════
# ETAPA 10 — IMPORTÂNCIA DAS VARIÁVEIS
# ══════════════════════════════════════════════════════════════════
print("\n[10/10] Análise de importância das variáveis...")

# Random Forest — Feature Importance
importancias_rf = pd.Series(rf.feature_importances_, index=feature_names)
top20_rf = importancias_rf.nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(11, 8))
cores_imp = [PALETA['acento'] if i >= 15 else PALETA['primario'] for i in range(len(top20_rf))]
bars = ax.barh(top20_rf.index, top20_rf.values, color=cores_imp,
               edgecolor='white', linewidth=1)
for bar, val in zip(bars, top20_rf.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8.5,
            fontweight='bold', color=PALETA['primario'])
ax.set_title('Random Forest — Top 20 Variáveis Mais Importantes', fontsize=13,
             fontweight='bold', color=PALETA['primario'], pad=15)
ax.set_xlabel('Importância (Impureza de Gini)', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT + '08_importancia_rf.png', dpi=150, bbox_inches='tight')
plt.close()

# Regressão Logística — Coeficientes (top 20 abs)
coef_lr = pd.Series(lr.coef_[0], index=feature_names)
top20_lr = coef_lr.abs().nlargest(20)
top20_lr_vals = coef_lr[top20_lr.index].sort_values()

fig, ax = plt.subplots(figsize=(11, 8))
cores_lr = [PALETA['acento'] if v > 0 else PALETA['positivo'] for v in top20_lr_vals.values]
ax.barh(top20_lr_vals.index, top20_lr_vals.values, color=cores_lr,
        edgecolor='white', linewidth=1)
ax.axvline(0, color=PALETA['cinza'], linewidth=1.5, linestyle='--')
ax.set_title('Regressão Logística — Top 20 Coeficientes (|valor|)', fontsize=13,
             fontweight='bold', color=PALETA['primario'], pad=15)
ax.set_xlabel('Coeficiente', fontsize=11)
patch_pos = mpatches.Patch(color=PALETA['acento'], label='Aumenta probabilidade de evasão')
patch_neg = mpatches.Patch(color=PALETA['positivo'], label='Reduz probabilidade de evasão')
ax.legend(handles=[patch_pos, patch_neg], fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT + '09_coeficientes_lr.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Gráficos de importância salvos")

# ══════════════════════════════════════════════════════════════════
# RELATÓRIO FINAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  RELATÓRIO FINAL")
print("=" * 65)
melhor = df_resultados.loc[df_resultados['F1-Score'].idxmax(), 'Modelo']
print(f"\n  Melhor modelo (F1-Score): {melhor}")
print(f"\n  Top 5 variáveis — Random Forest:")
top5 = importancias_rf.nlargest(5)
for var, imp in top5.items():
    print(f"    • {var:<45} {imp:.4f}")

print("\n  Top 5 variáveis — Regressão Logística (positivas):")
top5_lr_pos = coef_lr.nlargest(5)
for var, coef in top5_lr_pos.items():
    print(f"    • {var:<45} {coef:.4f}")

print(f"\n  → Todos os gráficos salvos em: {OUTPUT}")
print("\n✅  Pipeline concluído com sucesso!")
