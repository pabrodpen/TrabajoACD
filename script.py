import pandas as pd


# EJERCICIO 1

# Mostramos el dataset
df = pd.read_csv("resultados.csv")
print(df)

# EJERCICIO 2

# Exploración básica del DataFrame
print("Forma del DataFrame (filas, columnas):")
print(df.shape)

print("\nPrimeras 5 filas del DataFrame:")
print(df.head())

print("\nInformación general del DataFrame:")
print(df.info())

print("\nResumen estadístico de las columnas numéricas:")
print(df.describe())


# EJERCICIO 3

# Seleccionamos 4 columnas representativas
columnas_representativas = ["Gender", "Age", "MTRANS", "FAF"]
df_seleccion = df[columnas_representativas]

# Mostramos las primeras 5 filas de las columnas seleccionadas
print("\n")
print("Selección de 4 columnas representativas (5 primeras filas):")
print(df_seleccion.head())


# EJERICIO 4

# Añadir una nueva columna 'water_intake_per_meal' con la ingesta de agua por comida
df["water_intake_per_meal"] = df["CH2O"] / df["NCP"]

nuevas_columnas_representativas = [
    "Gender",
    "Age",
    "MTRANS",
    "FAF",
    "water_intake_per_meal",
]
df_seleccion_nueva_columna = df[nuevas_columnas_representativas]

print("\n")
print("Mostramos las columnas rerpesentativas + la nueva columna")
print(df_seleccion_nueva_columna.head())


# EJERCICIO 5

# Mostramos las filas donde el paciente es fumador y bebe frecuentemente
columnas_seleccion = [
    "Gender",
    "Age",
    "Height",
    "Weight",
]

fumadores_bebedores = df[(df["SMOKE"] == "yes") & (df["CALC"] == "Frequently")]
print("\nPersonas fumadoras y que consumen alchol de manera frecuente ")
print(fumadores_bebedores[columnas_seleccion])


# Crear un DataFrame con una fila de valores None
nueva_fila_nula = pd.DataFrame([{col: None for col in df.columns}])

# Concatenar la nueva fila al DataFrame original
df = pd.concat([df, nueva_fila_nula], ignore_index=True)

# Verificar y mostrar las filas con valores nulos
valores_nulos = df[df.isnull().any(axis=1)]
print("\nFilas con valores nulos:")
print(valores_nulos)

# Para cada fila con valores nulos, mostrar solo las columnas que son nulas
# for index, row in valores_nulos.iterrows():
#     print(f"Fila {index}:")
#     for column, value in row.items():
#         if pd.isnull(value):
#             print(f"  - {column} es nulo")


# Contamos el número de filas de nuestro csv
print(f"\nNumero de registros del csv: {df.shape[0]}")


# Borramos filas con valores nulos y mostramos el número luego de eliminarlas
df_limpio = df.dropna()  # Crear un nuevo DataFrame sin valores nulos
print(f"\nNumero de registros despues de eliminar las filas con valores nulos: {df_limpio.shape[0]}")


# Agrupamos por frecuencia de consumo de alcohol ("CALC") y tipo de obesidad ("OBESITY")
agrupacion = (
    df.groupby(["CALC", "NObeyesdad"]).size().reset_index(name="Numero de Personas")
)

# Mostrar resultados
print("\nNúmero de personas según la frecuencia de consumo de alcohol y tipo de obesidad:")
print(agrupacion)

# Promedio de peso por consumo de alimentos entre comidas (CAEC)
promedio_peso_por_caec = df.groupby("CAEC")[["Weight"]].mean()
print("\nPromedio de peso por consumo de alimentos entre comidas (CAEC):")
print(promedio_peso_por_caec)


