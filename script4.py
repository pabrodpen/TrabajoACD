import pandas as pd

# Cargar el dataset
df = pd.read_csv("resultados.csv")

# Aplicar One-Hot Encoding a todas las columnas categóricas excepto "NObeyesdad"
columnas_categoricas = [
    "Gender",
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

# Crear variables predictoras utilizando df.apply
# 1. Crear una nueva columna con la interacción entre la frecuencia de consumo de alimentos y el consumo de agua
df_encoded["Water_Food_Interaction"] = df_encoded.apply(
    lambda row: row["CH2O"] * row["FCVC"], axis=1
)

# 2. Crear una nueva columna con la proporción entre la altura y el peso
df_encoded["Height_Weight_Ratio"] = df_encoded.apply(
    lambda row: row["Height"] / row["Weight"], axis=1
)

# Crear una nueva variable combinando el historial familiar con el consumo de alimentos entre comidas
df_encoded["Family_Food_Interaction"] = df_encoded.apply(
    lambda row: (
        1
        if row["family_history_with_overweight_yes"] == 1
        and row["CAEC_Frequently"] == 1
        else 0
    ),
    axis=1,
)


# Calcular la correlación de todas las columnas con "NObeyesdad"
correlaciones = df_encoded.corr()["NObeyesdad"].sort_values(ascending=False)

# Mostrar las correlaciones
print("Correlación de cada columna con el tipo de obesidad:")
print(correlaciones)

# Mostrar las variables con alta correlación con NObeyesdad (> 0.9)
print("\nVariables predictoras con alta correlación con NObeyesdad:")
print(correlaciones[correlaciones > 0.9])
