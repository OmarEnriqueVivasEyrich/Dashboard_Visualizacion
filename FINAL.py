import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# Definir la ruta del archivo Parquet
file_path = 'DatosParquet_reducido.parquet'  # Cambiado a ruta relativa

# Configuración de estilo
st.set_page_config(page_title="Dashboard de Puntajes y Estratos", layout="wide")
st.title('Dashboard de Puntajes y Estratos por Departamento')

# Verificar si el archivo Parquet existe
# Cargar el archivo Parquet
df = pd.read_parquet(file_path)
st.dataframe(df.head())
