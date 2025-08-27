import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
)
from sklearn.dummy import DummyRegressor

sns.set(context='talk', style='ticks')

# ---------- Função de plot ----------
def plota_pivot_table(df, value, index, func, ylabel, xlabel, opcao='nada'):
    if opcao == 'nada':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).plot(figsize=[15, 5])
    elif opcao == 'unstack':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).unstack().plot(figsize=[15, 5])
    elif opcao == 'sort':
        # quando index é uma coluna única
        piv = pd.pivot_table(df, values=value, index=index, aggfunc=func)
        # se for MultiIndex, tenta ordenar pela primeira coluna resultante
        if isinstance(piv, pd.DataFrame) and value in piv.columns:
            piv.sort_values(value).plot(figsize=[15, 5])
        else:
            piv.sort_values(by=piv.columns.tolist()[0]).plot(figsize=[15, 5])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig=plt)

# ---------- Config da página ----------
st.set_page_config(
    page_title="Análise de Crédito",
    page_icon="https://images.vexels.com/media/users/3/129923/isolated/preview/23c69d5178087b811dd1196cf6274613-icone-de-cartoes-de-credito.png",
    layout="wide",
)
st.title("Análise exploratória e modelagem — Previsão de Renda ")

# ---------- Carregamento de dados ----------
df = pd.read_csv(r"C:/Users/Casa/Desktop/ebac/projeto2/projeto 2/previsao_de_renda.csv")

df = df.drop(columns=['Unnamed: 0', 'index'])
# Garante que data_ref é datetime para filtrar
if 'data_ref' in df.columns:
    if not np.issubdtype(df['data_ref'].dtype, np.datetime64):
        with st.spinner("Convertendo data_ref para datetime..."):
            df['data_ref'] = pd.to_datetime(df['data_ref'], errors='coerce')

# ---------- Pré-processamento ----------
cols_drop_modelo = ['Unnamed: 0', 'index']
for c in cols_drop_modelo:
    if c in df.columns:
        df = df.drop(columns=c)

# Limpeza (mantém a mesma lógica do teu script)
if 'tempo_emprego' in df.columns:
    df['tempo_emprego'] = df['tempo_emprego'].fillna(df['tempo_emprego'].median())
if 'qt_pessoas_residencia' in df.columns:
    df['qt_pessoas_residencia'] = df['qt_pessoas_residencia'].fillna(df['qt_pessoas_residencia'].median())

# Construção de features (condicional às colunas existirem)
if {'qtd_filhos', 'qt_pessoas_residencia'}.issubset(df.columns):
    df['densidade_dependentes'] = df['qtd_filhos'] / df['qt_pessoas_residencia']
if {'posse_de_imovel', 'posse_de_veiculo'}.issubset(df.columns):
    df['score_patrimonio'] = df[['posse_de_imovel', 'posse_de_veiculo']].sum(axis=1)
if {'educacao', 'tipo_renda'}.issubset(df.columns):
    df['educa_renda'] = df['educacao'].astype(str) + "_" + df['tipo_renda'].astype(str)
if {'tempo_emprego', 'idade'}.issubset(df.columns):
    # evita divisão por zero
    df['estabilidade'] = df['tempo_emprego'] / df['idade'].replace({0: np.nan})
    df['estabilidade'] = df['estabilidade'].fillna(0)

# Formatação de bool -> int
for col in ['posse_de_veiculo', 'posse_de_imovel', 'mau']:
    if col in df.columns and df[col].dtype == bool:
        df[col] = df[col].astype(int)

# ---------- SIDEBAR: filtros e controles ----------
st.sidebar.header("Filtros")

# Filtro por data
if 'data_ref' in df.columns and df['data_ref'].notna().any():
    min_data = pd.to_datetime(df['data_ref']).min()
    max_data = pd.to_datetime(df['data_ref']).max()

    data_inicial = st.sidebar.date_input(
        "Data inicial", value=min_data.date(), min_value=min_data.date(), max_value=max_data.date()
    )
    data_final = st.sidebar.date_input(
        "Data final", value=max_data.date(), min_value=min_data.date(), max_value=max_data.date()
    )

    # Aplica filtro
    mask = (df['data_ref'] >= pd.to_datetime(data_inicial)) & (df['data_ref'] <= pd.to_datetime(data_final))
    df = df.loc[mask].copy()

# Controles de gráfico pivot
st.sidebar.subheader("Gráfico (Pivot)")
# valor a ser agregado
valor_col = st.sidebar.selectbox(
    "Coluna de valor (numérica):",
    options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
    index=0
)
# índice (eixo x)
indice_cols = st.sidebar.multiselect(
    "Índice (uma ou mais colunas):",
    options=list(df.columns),
    default=['data_ref'] if 'data_ref' in df.columns else []
)
agg_func = st.sidebar.selectbox("Agregação:", options=['mean', 'median', 'sum', 'count'], index=0)
pivot_opcao = st.sidebar.selectbox("Opção de exibição:", options=['nada', 'unstack', 'sort'], index=0)
do_plot = st.sidebar.button("Gerar gráfico")

# ---------- EDA rápida ----------
st.subheader("Univariada ")
st.write(df.describe(include='all').T)

# Bivariada categórica
if 'mau' in df.columns:
    st.subheader("Bivariada — categorias vs. mau (mean)")
    bi_cat_cols = df.select_dtypes(include=["object", "bool"]).columns.drop("mau", errors="ignore")
    if len(bi_cat_cols) > 0:
        summary_bi_cat = (
            bi_cat_cols.to_series()
            .apply(lambda c: df.groupby(c)["mau"].mean().round(3).to_dict())
            .reset_index()
            .rename(columns={"index": "variavel", 0: "taxa_mau"})
        )
        st.write(summary_bi_cat)

# ---------- Plots Pivot opcionais ----------
if do_plot and len(indice_cols) > 0:
    try:
        idx = indice_cols if len(indice_cols) > 1 else indice_cols[0]
        plota_pivot_table(
            df=df, value=valor_col, index=idx, func=agg_func,
            ylabel=f"{agg_func} de {valor_col}", xlabel=str(idx), opcao=pivot_opcao
        )
    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")

# One-Hot Encoding das categóricas
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Para modelagem vamos remover 'data_ref' dos features
if 'data_ref' in cat_cols:
    cat_cols.remove('data_ref')

try:
    ohe = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
except TypeError:
    # fallback
    ohe = OneHotEncoder(sparse=False, drop=None, handle_unknown="ignore")

encoded = pd.DataFrame(
    ohe.fit_transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0)),
    columns=ohe.get_feature_names_out(cat_cols) if cat_cols else [],
    index=df.index
)

df_num = df.drop(columns=cat_cols)
# Remove 'data_ref' da modelagem
if 'data_ref' in df_num.columns:
    df_num = df_num.drop(columns=['data_ref'])

df_model = pd.concat([df_num, encoded], axis=1)

# ---------- Treino/Teste ----------
if 'renda' not in df_model.columns:
    st.error("A coluna 'renda' não foi encontrada para a modelagem.")
    st.stop()

X = df_model.drop(columns=["renda"])
y = df_model["renda"]

if X.empty:
    st.error("Não há features após o pré-processamento. Verifique os filtros/colunas.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Modelos ----------
modelos = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

# ---------- Avaliação ----------
resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)

    # RMSE compatível com versões antigas do sklearn
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, preds))

    resultados[nome] = {
        "R2": r2_score(y_test, preds),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_test, preds),
        "MedAE": median_absolute_error(y_test, preds)
    }

st.subheader("Resultados dos modelos")
st.write(pd.DataFrame(resultados).T)

# ---------- Importâncias (RandomForest) ----------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)

st.subheader("Top 15 Importâncias — RandomForest")
st.write(importances)

# ---------- Baseline ----------
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
preds_dummy = dummy.predict(X_test)
st.write("Baseline R²:", r2_score(y_test, preds_dummy))
