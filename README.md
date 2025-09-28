# Taller ML – Parte 1 (Housing)

Aplicación Streamlit para la **Parte 1** del taller: regresión para predecir `MEDV` usando `RM`, `LSTAT`, `PTRATIO`.

## Qué hace
- EDA: `.info()`, `.describe()`, nulos y distribuciones.
- **Modelos simples** (MEDV ~ RM, LSTAT, PTRATIO) con **recta, ecuación y R²**.
- **Regresión múltiple** (All-in) con ecuación, **R²**, métricas en **test** (R², RMSE, MAE) y **gráfico de residuales**.
- **Predicción interactiva** con sliders.

## Archivos
- `app.py`: código principal.
- `requirements.txt`: dependencias.
- `housing.csv` (opcional): puedes subirlo desde la app si no lo incluyes en el repo.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
