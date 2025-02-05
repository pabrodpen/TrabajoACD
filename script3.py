#de líneas, de dispersión, de barras,
#histograma y diagrama de cajas)
import matplotlib.pyplot as plt
import pandas as pd
# Mostramos el dataset
df = pd.read_csv("resultados.csv")
print(df)

#1.DIAGRAMA DE LINEAS

# Ordenar los datos por edad (opcional, para mejorar la visualización)
df = df.sort_values(by="Age")

# Crear el gráfico de líneas
plt.figure(figsize=(10, 5))  # Tamaño del gráfico
plt.plot(df["Age"], df["Weight"], marker='o', linestyle='-', color='b', label="Peso por Edad")

# Etiquetas y título
plt.xlabel("Edad (años)")
plt.ylabel("Peso (kg)")
plt.title("Relación entre Edad y Peso")
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Añadir cuadrícula para mejor visualización

# Mostrar el gráfico
plt.show()

#2.DIAGRAMA DE BARRAS
 #Contar el número de personas en cada categoría de CAEC
conteo_caec = df["CAEC"].value_counts()

# Crear gráfico de barras
plt.bar(conteo_caec.index, conteo_caec.values, color="orange", edgecolor="black")

# Etiquetas y título
plt.title("Número de Personas según Consumo de Alimentos Entre Comidas (CAEC)")
plt.xlabel("Categorías de Consumo (CAEC)")
plt.ylabel("Número de Personas")

# Mostrar el gráfico
plt.show()


#3.HISTOGRAMA
# Crear histograma de la columna 'Height'
plt.figure(figsize=(8, 5))  # Ajustar el tamaño del gráfico
plt.hist(df["Height"], bins=20, color="green", alpha=0.7, edgecolor="black")

# Etiquetas y título
plt.title("Distribución de la Estatura (Height)")
plt.xlabel("Altura (cm)")
plt.ylabel("Frecuencia")

# Mostrar el gráfico
plt.show()

#DIAGRAMA DE SECTORES
# Contar la frecuencia de cada medio de transporte
conteo_mtrans = df["MTRANS"].value_counts()

# Crear el gráfico de sectores
plt.figure(figsize=(8, 8))
plt.pie(conteo_mtrans.values, labels=conteo_mtrans.index, autopct="%1.1f%%", 
        startangle=90, colors=["lightblue", "orange", "green", "red", "purple"])
plt.title("Distribución de Medios de Transporte (MTRANS)")
plt.show()
