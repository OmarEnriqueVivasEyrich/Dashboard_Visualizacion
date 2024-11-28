import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de Streamlit
st.set_page_config(page_title="Dashboard Educativo", layout="wide")

# Cargar el archivo Parquet
@st.cache_data
def cargar_datos():
    return pd.read_parquet('DatosParquet.parquet')

df = cargar_datos()

# Título principal
st.title("Dashboard Educativo")

# Mostrar las primeras filas
st.header("Vista General de los Datos")
st.dataframe(df.head())

# Procesamiento y limpieza de datos
st.header("Procesamiento de Datos")
df_limpio = df[['ESTU_DEPTO_RESIDE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE',
                'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'FAMI_NUMLIBROS', 'PUNT_LECTURA_CRITICA',
                'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL']]

# Procesamiento de columnas
df_limpio['FAMI_ESTRATOVIVIENDA'] = df_limpio['FAMI_ESTRATOVIVIENDA'].replace({'Sin Estrato': None}).str.replace('Estrato ', '', regex=False).astype(float)
orden_educacion = [('Postgrado', 13), ('Educación profesional completa', 12), ('Educación profesional incompleta', 11),
                   ('Técnica o tecnológica completa', 10), ('Secundaria (Bachillerato) completa', 9),
                   ('Primaria completa', 8), ('Técnica o tecnológica incompleta', 7), ('Secundaria (Bachillerato) incompleta', 6),
                   ('Primaria incompleta', 5), ('Ninguno', 4), ('No Aplica', 3), ('No sabe', 2), (None, 1)]
diccionario_educacion = dict(orden_educacion)
df_limpio['FAMI_EDUCACIONPADRE'] = df_limpio['FAMI_EDUCACIONPADRE'].replace(diccionario_educacion)
df_limpio['FAMI_EDUCACIONMADRE'] = df_limpio['FAMI_EDUCACIONMADRE'].replace(diccionario_educacion)
df_limpio['FAMI_TIENEINTERNET'] = df_limpio['FAMI_TIENEINTERNET'].replace({'Sí': 1, 'No': 0, 'Si': 1}).astype(float)
df_limpio['FAMI_TIENECOMPUTADOR'] = df_limpio['FAMI_TIENECOMPUTADOR'].replace({'Sí': 1, 'No': 0, 'Si': 1}).astype(float)
orden_libros = [('MÁS DE 100 LIBROS', 5), ('26 A 100 LIBROS', 4), ('11 A 25 LIBROS', 3), ('0 A 10 LIBROS', 2), (None, 1)]
diccionario_libros = dict(orden_libros)
df_limpio['FAMI_NUMLIBROS'] = df_limpio['FAMI_NUMLIBROS'].replace(diccionario_libros).astype(float)

st.write("Datos procesados:")
st.dataframe(df_limpio.head())

# Selección de puntaje
puntaje_opciones = ['PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL']
puntaje_seleccionado = st.selectbox("Seleccione el puntaje para analizar:", puntaje_opciones)

# Análisis por departamento
df_agrupado = df_limpio.groupby('ESTU_DEPTO_RESIDE')[puntaje_seleccionado].mean().reset_index()

mejor_departamento = df_agrupado.loc[df_agrupado[puntaje_seleccionado].idxmax()]
peor_departamento = df_agrupado.loc[df_agrupado[puntaje_seleccionado].idxmin()]
df_comparacion = pd.DataFrame([mejor_departamento, peor_departamento])

# Visualización
st.header("Visualización de Datos")
st.subheader(f"Comparativa del {puntaje_seleccionado}: Mejor vs Peor Departamento")
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))
bar_plot = sns.barplot(data=df_comparacion, y='ESTU_DEPTO_RESIDE', x=puntaje_seleccionado, palette=['#006400', '#8B0000'])
plt.title(f'Comparativa del {puntaje_seleccionado.replace("_", " ")}', fontsize=18, weight='bold', color='black')
plt.xlabel(f'Media del {puntaje_seleccionado.replace("_", " ")}', fontsize=16, fontweight='bold')
plt.ylabel('Departamento', fontsize=16, fontweight='bold')
for p in bar_plot.patches:
    value = round(p.get_width())
    bar_plot.annotate(f'{value}', (p.get_width() / 2, p.get_y() + p.get_height() / 2), ha='center', color='white', weight='bold', fontsize=12)
st.pyplot(plt)

# Final
st.write("Análisis completado.")
