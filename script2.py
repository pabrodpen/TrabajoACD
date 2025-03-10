import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el dataset
df = pd.read_csv("resultados.csv")  # Cambia el nombre según corresponda

# Aplicar One-Hot Encoding a la columna categórica
df_encoded = pd.get_dummies(df, columns=['obesity_level'], drop_first=True)

# Mostrar las nuevas columnas generadas
print("Columnas después de One-Hot Encoding:", df_encoded.columns)

# Seleccionar la variable independiente (peso)
X = df[['weight']].values  # Variable independiente (peso)

# Seleccionar una categoría específica como variable dependiente (ejemplo: "Obesidad Tipo I")
y = df_encoded[['obesity_level_Obesity Type I']].values  # 1 si es Obesidad Tipo I, 0 si no

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Hacer predicciones
y_pred = model.predict(X)

# Mostrar coeficientes
print("Intercepto (w₀):", model.intercept_[0])
print("Pendiente (w₁):", model.coef_[0][0])

# Graficar la regresión
plt.scatter(X, y, color='blue', label='Datos reales')  # Puntos originales
plt.plot(X, y_pred, color='red', label='Línea de regresión')  # Línea de regresión
plt.title('Regresión Lineal Simple: Peso vs. Obesidad Tipo I')
plt.xlabel('Peso (kg)')
plt.ylabel('Probabilidad de Obesidad Tipo I')
plt.legend()
plt.grid(True)
plt.show()
