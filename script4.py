import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("resultados.csv")

# Convertir las columnas a tipo numérico, forzando la conversión de errores a NaN
df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Convertir 'Gender' a valores numéricos
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Aplicar One-Hot Encoding a todas las columnas categóricas excepto "NObeyesdad"
columnas_categoricas = [
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]
df_encoded = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)

# Convertir la variable objetivo "NObeyesdad" a valores numéricos
obesity_mapping = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6,
}
df_encoded["NObeyesdad"] = df["NObeyesdad"].map(obesity_mapping)

# Calcular las variables IMC, PBF y REE
df_encoded["IMC"] = df_encoded["Weight"] / (df_encoded["Height"] ** 2)
df_encoded["PBF"] = (
    (1.20 * (df_encoded["Weight"] / (df_encoded["Height"] ** 2)))
    + (0.23 * df_encoded["Age"])
    - (10.8 * df_encoded["Gender"])
    - 5.4
)
df_encoded["REE"] = (
    (10 * df_encoded["Weight"])
    + (6.25 * df_encoded["Height"] * 100)
    - (5 * df_encoded["Age"])
    + (5 * df_encoded["Gender"] - 161 * (1 - df_encoded["Gender"]))
)

# Calcular la correlación de todas las variables con NObeyesdad
correlaciones = df_encoded.corr()["NObeyesdad"].sort_values(ascending=False)

# Mostrar resultados
print("\n📊 Correlación de todas las variables con el tipo de obesidad:")
print(correlaciones)


# Identificación de valores nulos en las variables
print("\n📌 Valores nulos antes de la imputación:")
print(df_encoded[["IMC", "Weight", "PBF", "REE"]].isnull().sum())

# Imputar valores nulos con la **mediana**
for col in ["IMC", "Weight", "PBF", "REE"]:
    df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

print("\n✅ Valores nulos después de la imputación:")
print(df_encoded[["IMC", "Weight", "PBF", "REE"]].isnull().sum())

# Identificación de errores en los datos
errores = {
    "Weight": df_encoded["Weight"] < 0,  # Peso negativo
    "IMC": df_encoded["IMC"] < 10,  # IMC poco realista
    "PBF": df_encoded["PBF"] < 0,  # Porcentaje de grasa corporal negativo
    "REE": df_encoded["REE"] < 0,  # Tasa metabólica en reposo negativa
}

# Mostrar registros con errores
for columna, condicion in errores.items():
    print(f"\n❌ Errores en la columna {columna}:")
    print(df_encoded[condicion])

# Eliminamos registros con datos erróneos
df_encoded = df_encoded[~df_encoded["Weight"].lt(0)]
df_encoded = df_encoded[~df_encoded["IMC"].lt(10)]
df_encoded = df_encoded[~df_encoded["PBF"].lt(0)]
df_encoded = df_encoded[~df_encoded["REE"].lt(0)]

# Visualización de outliers con boxplots
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

columnas_outliers = ["Weight", "IMC", "PBF", "REE"]
colores = ["lightblue", "lightcoral", "lightgreen", "lightyellow"]

for i, col in enumerate(columnas_outliers):
    sns.boxplot(y=df_encoded[col], ax=axes[i], color=colores[i])
    axes[i].set_title(f"Valores atípicos en {col}")

plt.tight_layout()
plt.show()

# Tratamiento de outliers: Sustitución con la mediana
for col in columnas_outliers:
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Sustituir valores atípicos con la mediana
    mediana_valor = df_encoded[col].median()
    df_encoded.loc[df_encoded[col] < limite_inferior, col] = mediana_valor
    df_encoded.loc[df_encoded[col] > limite_superior, col] = mediana_valor

print("\n✅ Outliers corregidos con la mediana.")

# Mostrar el DataFrame limpio
print("\n✅ Dataset final después de la limpieza:")
print(df_encoded.head())
