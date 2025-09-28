# app.py
# ============================================
# Taller ML – Parte 1: Regresión (Housing)
# ============================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Taller ML – Housing", page_icon="🏠", layout="wide")
sns.set_style("whitegrid")

# ---------- Helpers ----------
@st.cache_data
def read_csv_any(file):
    if isinstance(file, str):
        return pd.read_csv(file)
    return pd.read_csv(file)

def df_info_str(df):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def scatter_with_fit(df, xcol, ycol="MEDV", sharey=None):
    X = df[[xcol]].values
    y = df[ycol].values
    lr = LinearRegression().fit(X, y)
    y_hat = lr.predict(X)
    r2 = r2_score(y, y_hat)
    b0, b1 = lr.intercept_, lr.coef_[0]

    order = np.argsort(X.ravel())
    fig, ax = plt.subplots()
    sns.scatterplot(x=xcol, y=ycol, data=df, ax=ax, alpha=0.6)
    ax.plot(X.ravel()[order], y_hat[order])
    title = f"{xcol} vs {ycol} — R²={r2:.3f}\n{ycol} = {b0:,.0f} + ({b1:,.2f})·{xcol}"
    ax.set_title(title)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.grid(alpha=0.25)
    if sharey: ax.set_ylim(sharey)
    st.pyplot(fig)
    return r2, b0, b1

# ---------- Sidebar ----------
st.sidebar.title("Configuración")
uploaded = st.sidebar.file_uploader("Sube housing.csv (opcional)", type=["csv"])
st.sidebar.markdown(
    "El dataset debe tener las columnas: **RM**, **LSTAT**, **PTRATIO**, **MEDV**."
)

# Carga de datos
df = None
if uploaded is not None:
    df = read_csv_any(uploaded)
else:
    # Si subes el CSV al repo, descomenta la línea de abajo:
    # df = read_csv_any("housing.csv")
    pass

st.title("🏠 Taller ML – Parte 1: Regresión (Housing)")

if df is None:
    st.info("Sube **housing.csv** en la barra lateral o coloca el archivo junto a *app.py* y descomenta la línea en el código.")
    st.stop()

# Validación de columnas
needed = ["RM", "LSTAT", "PTRATIO", "MEDV"]
if not all(c in df.columns for c in needed):
    st.error(f"Faltan columnas. Se requieren: {needed}. Encontradas: {df.columns.tolist()}")
    st.stop()

# ---------- Sección 1: EDA ----------
st.header("1) Exploración de datos")
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Vista rápida")
    st.dataframe(df.head(10))
    st.markdown("**.describe()**")
    st.dataframe(df[needed].describe().round(3))

with c2:
    st.subheader("Información del DataFrame")
    st.code(df_info_str(df))
    st.markdown("**Valores nulos**")
    st.dataframe(df[needed].isna().sum().to_frame("nulos"))

st.subheader("Distribuciones")
c3, c4 = st.columns(2)
with c3:
    fig, ax = plt.subplots()
    sns.histplot(df["RM"], kde=False, ax=ax); ax.set_title("Distribución RM"); st.pyplot(fig)
    fig, ax = plt.subplots()
    sns.histplot(df["LSTAT"], kde=False, ax=ax); ax.set_title("Distribución LSTAT"); st.pyplot(fig)
with c4:
    fig, ax = plt.subplots()
    sns.histplot(df["PTRATIO"], kde=False, ax=ax); ax.set_title("Distribución PTRATIO"); st.pyplot(fig)
    fig, ax = plt.subplots()
    sns.histplot(df["MEDV"], kde=False, ax=ax); ax.set_title("Distribución MEDV"); st.pyplot(fig)

# ---------- Sección 2: Modelos lineales simples ----------
st.header("2) Modelos lineales simples (MEDV ~ X)")
ymin, ymax = df["MEDV"].min()*0.9, df["MEDV"].max()*1.05
colA, colB, colC = st.columns(3)

with colA: r2_rm, b0_rm, b1_rm = scatter_with_fit(df, "RM", "MEDV", sharey=(ymin, ymax))
with colB: r2_ls, b0_ls, b1_ls = scatter_with_fit(df, "LSTAT", "MEDV", sharey=(ymin, ymax))
with colC: r2_pt, b0_pt, b1_pt = scatter_with_fit(df, "PTRATIO", "MEDV", sharey=(ymin, ymax))

st.markdown("**Comparación R² (simple):**")
st.dataframe(
    pd.DataFrame({
        "Variable": ["LSTAT", "RM", "PTRATIO"],
        "R²": [r2_ls, r2_rm, r2_pt],
        "Ecuación": [
            f"MEDV = {b0_ls:,.0f} + ({b1_ls:,.2f})·LSTAT",
            f"MEDV = {b0_rm:,.0f} + ({b1_rm:,.2f})·RM",
            f"MEDV = {b0_pt:,.0f} + ({b1_pt:,.2f})·PTRATIO",
        ]
    }).sort_values("R²", ascending=False)
)

# ---------- Sección 3: Regresión múltiple ----------
st.header("3) Regresión lineal múltiple (All-in)")
X = df[["RM", "LSTAT", "PTRATIO"]].values
y = df["MEDV"].values

# Ecuación en todo el set (solo para mostrar coeficientes)
lr_all = LinearRegression().fit(X, y)
y_hat_all = lr_all.predict(X)
r2_all = r2_score(y, y_hat_all)
coef_map = dict(zip(["RM", "LSTAT", "PTRATIO"], lr_all.coef_))
st.markdown(
    f"**Ecuación:** `MEDV = {lr_all.intercept_:,.0f} + "
    f"({coef_map['RM']:.2f})·RM + ({coef_map['LSTAT']:.2f})·LSTAT + ({coef_map['PTRATIO']:.2f})·PTRATIO`"
)
st.markdown(f"**R² (entrenamiento completo):** `{r2_all:.3f}`")

# Split para métricas de generalización
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
lr = LinearRegression().fit(X_tr, y_tr)
y_pred = lr.predict(X_te)
r2_te = r2_score(y_te, y_pred)
rmse = mean_squared_error(y_te, y_pred, squared=False)
mae = mean_absolute_error(y_te, y_pred)

st.markdown("**Métricas en test (25%)**")
st.dataframe(pd.DataFrame([{"R²_test": r2_te, "RMSE": rmse, "MAE": mae}]))

# Residuales
fig, ax = plt.subplots()
ax.scatter(y_pred, y_te - y_pred, alpha=0.6)
ax.axhline(0, linestyle="--", color="gray")
ax.set_xlabel("Predicción (test)")
ax.set_ylabel("Residual")
ax.set_title("Residuales vs Predicción — Regresión múltiple")
st.pyplot(fig)

# ---------- Sección 4: Predicción interactiva ----------
st.header("4) Predicción interactiva (modelo múltiple)")
c1, c2, c3 = st.columns(3)
rm = c1.slider("RM (habitaciones promedio)", float(df.RM.min()), float(df.RM.max()), float(df.RM.median()))
lstat = c2.slider("LSTAT (% estatus bajo)", float(df.LSTAT.min()), float(df.LSTAT.max()), float(df.LSTAT.median()))
ptr = c3.slider("PTRATIO (alumnos/profesor)", float(df.PTRATIO.min()), float(df.PTRATIO.max()), float(df.PTRATIO.median()))

pred = lr.predict(np.array([[rm, lstat, ptr]]))[0]
st.metric("Valor estimado de la vivienda (USD)", f"${pred:,.0f}")

st.caption(
    "Notas: Los modelos simples muestran la fuerza de relación individual (R²). "
    "El modelo múltiple combina predictores y suele mejorar el poder explicativo. "
    "El gráfico de residuales apoya la evaluación de linealidad/homocedasticidad."
)
