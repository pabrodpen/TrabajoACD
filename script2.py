import pandas as pd
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib.pyplot as plt




# Mostramos el dataset
df = pd.read_csv("resultados.csv")
print(df)




# EJERCICIO 1
print("EJERCICIO 1")


# Calcular el rango intercuartil (IQR) para la columna height
print("Primer cuartil:")
q1 = df["Age"].quantile(0.25)  # Primer cuartil
print(q1)
print("Segundo cuartil:")
q3 = df["Age"].quantile(0.75)  # Tercer cuartil
print(q3)
iqr = q3 - q1  # Rango intercuartil


# Definir los límites inferior y superior
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"Límite inferior: {lower_bound}, Límite superior: {upper_bound}")


# Filtrar valores dentro de los límites
df_sin_atipicos = df[(df["Age"] >= lower_bound) & (df["Age"] <= upper_bound)]
# Comparar el número de filas antes y después
print(f"Filas antes: {df.shape[0]}")
print(f"Filas después de eliminar atípicos: {df_sin_atipicos.shape[0]}")




# EJERCICIO 2
print("EJERCICIO 2")


# Calcular los valores mínimo y máximo de la columna 'Height'
min_height = df["Height"].min()
max_height = df["Height"].max()


print(f"Mínimo altura: {min_height}")
print(f"Máximo altura: {max_height}")


# Normalizar la columna 'Height'
df["Height_norm"] = (df["Height"] - min_height) / (max_height - min_height)
# Mostrar los resultados
print(df[["Height", "Height_norm"]].head())




# EJERCICIO 3
print("EJERCICIO 3")

# Aplicar One-Hot Encoding a la columna "CALC (Consumo de alcohol)"
df_encoded = pd.get_dummies(df, columns=["CALC"], drop_first=False)



# Mostrar un ejemplo de los datos codificados
print("\nDatos después de aplicar One-Hot Encoding:")
print(df_encoded.head())




# EJERCICIO 4
print("EJERCICIO 4")
# Separar las variables predictoras (X) y objetivo (y)
X = df[["Gender","Age","Height","Weight","family_history_with_overweight","FAVC","FCVC","NCP",
"CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS"]]  # Columnas predictoras
y = df["NObeyesdad"]  # Variable objetivo


# Mostrar las primeras filas de las columnas predictoras (X)
print("Primeras filas de las columnas predictoras (X):")
print(X.head())


# Mostrar las primeras filas de la variable objetivo (y)
print("\nPrimeras filas de la variable objetivo (y):")
print(y.head())




# EJERCICIO 5
print("EJERCICIO 5")


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)


print("Tamaño del conjunto de entrenamiento:", X_train.shape[0])
print("Tamaño del conjunto de prueba:", X_test.shape[0])


# Mostrar las primeras filas de cada conjunto de datos
print("\nPrimeras filas del conjunto de entrenamiento (X_train):")
print(X_train.head())


print("\nPrimeras filas del conjunto de prueba (X_test):")
print(X_test.head())


print(
   "\nPrimeras filas de la variable objetivo del conjunto de entrenamiento (y_train):"
)
print(y_train.head())


print("\nPrimeras filas de la variable objetivo del conjunto de prueba (y_test):")
print(y_test.head())




# EJERCICIO 6
print("EJERCICIO 6")


# Convertir la altura a centímetros
df["Height_cm"] = df["Height"] * 100




# Crear una nueva columna 'IMC' (Índice de Masa Corporal)
def calcular_imc(row):
   height_in_meters = row["Height_cm"] / 100  # Convertir altura de cm a metros
   return row["Weight"] / height_in_meters**2




df["IMC"] = df.apply(calcular_imc, axis=1)




# Mostrar un ejemplo de los datos con la nueva columna 'IMC' y 'Height_cm'
print("\nDatos con la nueva columna 'IMC' y 'Height_cm':")
print(df[["Gender", "Age", "Height", "Height_cm", "Weight", "IMC"]].head())


# EJERCICIO 7


# Dataset relacionado con beneficios
df_data = {
   "FCVC": [1, 2, 3],
   "Beneficios": ["Mala digestión", "Salud moderada", "Salud óptima"],
}
df_nuevo = pd.DataFrame(df_data)


# Unión de DataFrames por la columna "FCVC"
df_merged = df.merge(df_nuevo, on="FCVC", how="left")


# Mostrar el resultado (incluyendo la columna FCVC)
print(df_merged[["FCVC", "Beneficios"] + [col for col in df.columns]])



