import streamlit as st
import pandas as pd

# Título de la aplicación
st.title("Visualizador de archivos Parquet")

# Leer el archivo Parquet
try:
    # Cambia la ruta al archivo si está en otro lugar
    df = pd.read_parquet('DatosParquet.parquet')
    st.write("Archivo cargado exitosamente!")
    
    # Mostrar las primeras filas del DataFrame
    st.write("Primeras filas del DataFrame:")
    st.dataframe(df.head())  # Mostrar solo las primeras 5 filas
    
except Exception as e:
    st.error(f"Error al cargar el archivo Parquet: {e}")
