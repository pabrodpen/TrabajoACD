import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder  # Para convertir la variable categórica

# Cargar el dataset
df = pd.read_csv("resultados.csv")  

# Verificar nombres de columnas
print("\nColumnas en el dataset:", df.columns)

# Asegurar que las columnas no tengan espacios extra
df.columns = df.columns.str.strip()

# Verificar valores únicos en la columna 'NObeyesdad'
print("\nValores únicos en 'NObeyesdad':", df["NObeyesdad"].unique())

# Convertir la variable categórica 'NObeyesdad' a números
encoder = LabelEncoder()
df["NObeyesdad_encoded"] = encoder.fit_transform(df["NObeyesdad"])

# Mostrar cómo se codificaron las categorías
print("\nConversión de categorías:")
for categoria, codigo in zip(encoder.classes_, encoder.transform(encoder.classes_)):
    print(f"{categoria}: {codigo}")

# Seleccionar variables
X = df[['Weight']].values  # Variable independiente (peso)
y = df[['NObeyesdad_encoded']].values  # Variable dependiente (nivel de obesidad en números)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Hacer predicciones
y_pred = model.predict(X)

# Mostrar coeficientes
print("\nIntercepto (w₀):", model.intercept_[0])
print("Pendiente (w₁):", model.coef_[0][0])

# Graficar la regresión
plt.scatter(X, y, color='blue', label='Datos reales')  # Puntos originales
plt.plot(X, y_pred, color='red', label='Línea de regresión')  # Línea de regresión
plt.title('Regresión Lineal Simple: Peso vs. Nivel de Obesidad (Codificado)')
plt.xlabel('Peso (kg)')
plt.ylabel('Nivel de Obesidad')
plt.legend()
plt.grid(True)
plt.show()


#REGRESION MULTIPLE

# 📌 2. Calcular IMC y PBF
df["IMC"] = df["Weight"] / (df["Height"] ** 2)

# Convertir Gender a numérico: 1 para hombres, 0 para mujeres
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})  

df["PBF"] = (
    (1.20 * df["IMC"]) +
    (0.23 * df["Age"]) -
    (10.8 * df["Gender"]) -
    5.4
)

# 📌 4. Seleccionar variables para la regresión
X = df[['Weight', 'IMC', 'PBF']].values  # Variables independientes
y = df['NObeyesdad_encoded'].values  # Variable dependiente

# 📌 5. Crear el modelo y ajustarlo a los datos
model = LinearRegression()
model.fit(X, y)

# Hacer predicciones
y_pred = model.predict(X)

# 📌 6. Mostrar coeficientes
print("Intercepto (w₀):", model.intercept_)
print("Coeficientes (w₁, w₂, w₃):", model.coef_)

# 📌 7. Mostrar pares de puntos
print("\nPares de puntos (Weight, IMC, PBF, NObeyesdad):")
for (w, imc, pbf), yi in zip(X, y):
    print(f"({w:.2f}, {imc:.2f}, {pbf:.2f}, {yi:.2f})")

# 📌 8. Visualización de la regresión con planos separados (solo con 2 variables independientes)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Puntos originales
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Datos originales')

# Crear una malla para la superficie de predicción
x1_grid, x2_grid = np.meshgrid(
    np.linspace(min(X[:, 0]), max(X[:, 0]), 10),
    np.linspace(min(X[:, 1]), max(X[:, 1]), 10)
)

# Calcular los valores predichos en la malla usando solo Weight e IMC
y_pred_grid = (model.intercept_ +
               model.coef_[0] * x1_grid +
               model.coef_[1] * x2_grid)

# Dibujar la superficie de predicción
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color='red', alpha=0.5)

# Etiquetas y título
ax.set_title('Regresión Lineal Múltiple: Peso, IMC y Nivel de Obesidad')
ax.set_xlabel('Peso (kg)')
ax.set_ylabel('IMC')
ax.set_zlabel('Nivel de Obesidad')
ax.legend(["Datos originales", "Plano de predicciones"])

plt.show()