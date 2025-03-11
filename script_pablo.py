import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from math import trunc

# Cargar el dataset
df = pd.read_csv("resultados.csv")


# Convertir variables categóricas a numéricas
encoder = LabelEncoder()
df["NObeyesdad_encoded"] = encoder.fit_transform(df["NObeyesdad"])


#  Cálculo del IMC
df["IMC"] = df["Weight"] / (df["Height"] ** 2)

#  REGRESIÓN LINEAL SIMPLE (Weight → Obesidad)

# Seleccionar variables
X = df[["Weight"]].values
y = df[["NObeyesdad_encoded"]].values

# División en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Mostrar coeficientes
print("\n REGRESIÓN LINEAL SIMPLE")
print("Intercepto:", model.intercept_[0])
print("Pendiente:", model.coef_[0][0])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse = truncate(rmse, 2)
print("Error cuadrático medio (RMSE):", rmse)

# Graficar la regresión sobre 30 datos aleatorios de prueba
indices_aleatorios = np.random.choice(len(X_test), 30, replace=False)
plt.scatter(
    X_test[indices_aleatorios],
    y_test[indices_aleatorios],
    color="blue",
    label="Datos reales",
)
plt.plot(
    X_test[indices_aleatorios],
    y_pred[indices_aleatorios],
    color="red",
    label="Línea de regresión",
)
plt.title("Regresión Lineal Simple: Peso vs. Nivel de Obesidad (30 datos)")
plt.xlabel("Peso (kg)")
plt.ylabel("Nivel de Obesidad")
plt.legend()
plt.grid(True)
plt.show()


###  REGRESIÓN LINEAL MÚLTIPLE ###
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

df["PBF"] = (
    (1.20 * (df["Weight"] / (df["Height"] ** 2)))
    + (0.23 * df["Age"])
    - (10.8 * df["Gender"])
    - 5.4
)

# Seleccionar variables
X = df[["Weight", "IMC", "PBF"]].values
y = df["NObeyesdad_encoded"].values

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Mostrar coeficientes
print("\n REGRESIÓN LINEAL MÚLTIPLE")
print("Intercepto:", model.intercept_)
print("Coeficientes:", model.coef_)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse = truncate(rmse, 2)

print("Error cuadrático medio (RMSE):", rmse)

# Graficar la regresión sobre 30 datos aleatorios de prueba (solo Weight e IMC para 3D)
indices_aleatorios = np.random.choice(len(X_test), 30, replace=False)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Puntos de datos reales
ax.scatter(
    X_test[indices_aleatorios, 0],  # Peso
    X_test[indices_aleatorios, 1],  # IMC
    y_test[indices_aleatorios],  # Nivel de obesidad real
    color="blue",
    label="Datos reales",
)

#  Crear la malla para el plano de regresión
peso_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 10)
imc_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 10)
peso_mesh, imc_mesh = np.meshgrid(peso_range, imc_range)

# Calcular los valores del plano de regresión (predicción)
obesidad_pred_mesh = (
    model.intercept_ + model.coef_[0] * peso_mesh + model.coef_[1] * imc_mesh
)

# Dibujar el plano de regresión en color rojo
ax.plot_surface(peso_mesh, imc_mesh, obesidad_pred_mesh, color="red", alpha=0.5)

# Etiquetas
ax.set_title("Regresión Lineal Múltiple con Plano de Regresión")
ax.set_xlabel("Peso (kg)")
ax.set_ylabel("IMC")
ax.set_zlabel("Nivel de Obesidad")
ax.legend(["Datos originales", "Plano de regresión"])

plt.show()

### 3. REGRESIÓN XGBOOST ###

# Seleccionar variables predictoras y variable objetivo
X_xgb = df[["IMC", "Weight", "PBF"]]
y_xgb = df["NObeyesdad_encoded"]

# División en conjunto de entrenamiento (80%) y prueba (20%)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42
)

# Crear y entrenar el modelo XGBoost
model_xgb = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
)
model_xgb.fit(X_train_xgb, y_train_xgb)

# Hacer predicciones
y_pred_xgb = model_xgb.predict(X_test_xgb)
# Calcular RMSE
rmse_xgb = np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb))
rmse = truncate(rmse, 2)
print(f" RMSE de Regresión XGBoost: {rmse_xgb:.2f}")

#  Graficar la regresión sobre 30 datos aleatorios de prueba
indices_aleatorios = np.random.choice(len(X_test_xgb), 30, replace=False)

plt.figure(figsize=(10, 5))

# Dibujar los valores reales como puntos azules
plt.scatter(
    range(1, 31),
    y_test_xgb.iloc[indices_aleatorios],
    color="blue",
    label="Valores reales",
)

# Dibujar la línea de predicciones en naranja
plt.plot(
    range(1, 31),
    y_pred_xgb[indices_aleatorios],
    color="orange",
    linestyle="-",
    marker="o",
    label="Predicciones",
)

# Configuración de ejes y título
plt.xlabel("Registros (1-30)")
plt.ylabel("Nivel de Obesidad (Codificado)")
plt.title("Regresión XGBoost: Valores Reales vs. Predichos (30 datos)")
plt.legend()
plt.grid(True)
plt.show()


# REGRESION LINEAL SIMPLE SIN NULOS
# Imputación de valores nulos
df["Weight"].fillna(df["Weight"].median(), inplace=True)
df["Height"].fillna(df["Height"].median(), inplace=True)
df["family_history_with_overweight"].fillna("no", inplace=True)

# REGRESIÓN LINEAL SIMPLE SIN OUTLIERS

# Identificar outliers en "Weight" usando IQR
Q1 = df["Weight"].quantile(0.25)
Q3 = df["Weight"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# 📌 Reemplazar outliers por la mediana en lugar de eliminarlos
mediana_weight = df["Weight"].median()
df.loc[df["Weight"] < limite_inferior, "Weight"] = mediana_weight
df.loc[df["Weight"] > limite_superior, "Weight"] = mediana_weight

# Seleccionar variables
X = df[["Weight"]].values
y = df[["NObeyesdad_encoded"]].values

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Mostrar coeficientes
print("\n📌 REGRESIÓN LINEAL SIMPLE (Sin Outliers)")
print("Intercepto:", model.intercept_[0])
print("Pendiente:", model.coef_[0][0])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse = truncate(rmse, 2)
print("Error cuadrático medio (RMSE):", rmse)

# 📌 Graficar la regresión sobre 30 datos aleatorios de prueba
indices_aleatorios = np.random.choice(len(X_test), 30, replace=False)
plt.scatter(
    X_test[indices_aleatorios],
    y_test[indices_aleatorios],
    color="blue",
    label="Datos reales",
)
plt.plot(
    X_test[indices_aleatorios],
    y_pred[indices_aleatorios],
    color="red",
    label="Línea de regresión",
)
plt.title("Regresión Lineal Simple (Sin Outliers)")
plt.xlabel("Peso (kg)")
plt.ylabel("Nivel de Obesidad")
plt.legend()
plt.grid(True)
plt.show()


############
