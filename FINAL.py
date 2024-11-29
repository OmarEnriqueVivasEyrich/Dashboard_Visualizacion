import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# Definir la ruta del archivo Parquet
file_path = 'parquet.parquet'  # Cambiado a ruta relativa

# Configuración de estilo
st.set_page_config(page_title="Dashboard de Puntajes y Estratos", layout="wide")
st.title('Dashboard de Puntajes y Estratos por Departamento')

# Verificar si el archivo Parquet existe
if not os.path.exists(file_path):
    st.error(f"El archivo {file_path} no existe. Por favor, verifique la ruta.")
else:
    try:
        # Cargar el archivo Parquet
        df = pd.read_parquet(file_path)
        st.dataframe(df.head())

        # Mostrar los nombres de las columnas
        st.write("Nombres de las columnas en el DataFrame:")
        st.write(df.columns.tolist())

        # Procesamiento previo de datos
        df_limpio = df[['ESTU_DEPTO_RESIDE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE',
                         'FAMI_EDUCACIONMADRE', 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 
                         'FAMI_NUMLIBROS', 'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 
                         'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL']]

        df_limpio['FAMI_ESTRATOVIVIENDA'] = df_limpio['FAMI_ESTRATOVIVIENDA'].replace({
            'Sin Estrato': None}).str.replace('Estrato ', '', regex=False).astype(float)

        diccionario_educacion = {
            'Postgrado': 13,
            'Educación profesional completa': 12,
            'Educación profesional incompleta': 11,
            'Técnica o tecnológica completa': 10,
            'Secundaria (Bachillerato) completa': 9,
            'Primaria completa': 8,
            'Técnica o tecnológica incompleta': 7,
            'Secundaria (Bachillerato) incompleta': 6,
            'Primaria incompleta': 5,
            'Ninguno': 4,
            'No Aplica': 3,
            'No sabe': 2,
            None: 1
        }

        df_limpio['FAMI_EDUCACIONPADRE'] = df_limpio['FAMI_EDUCACIONPADRE'].replace(diccionario_educacion)
        df_limpio['FAMI_EDUCACIONMADRE'] = df_limpio['FAMI_EDUCACIONMADRE'].replace(diccionario_educacion)
        
        df_limpio['FAMI_TIENEINTERNET'] = df_limpio['FAMI_TIENEINTERNET'].replace({'Sí': 1, 'No': 0, 'Si': 1}).astype(float)
        df_limpio['FAMI_TIENECOMPUTADOR'] = df_limpio['FAMI_TIENECOMPUTADOR'].replace({'Sí': 1, 'No': 0, 'Si': 1}).astype(float)

        diccionario_libros = {
            'MÁS DE 100 LIBROS': 5,
            '26 A 100 LIBROS': 4,
            '11 A 25 LIBROS': 3,
            '0 A 10 LIBROS': 2,
            None: 1
        }

        df_limpio['FAMI_NUMLIBROS'] = df_limpio['FAMI_NUMLIBROS'].replace(diccionario_libros).astype(float)

        st.write("DataFrame limpio:")
        st.dataframe(df_limpio)

        # Lista de columnas de puntajes disponibles
        puntaje_opciones = [
            'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 
            'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL'
        ]

        puntaje_seleccionado = st.selectbox("Seleccione el puntaje para realizar el gráfico:", puntaje_opciones)
        st.write(f"Ha seleccionado: {puntaje_seleccionado}")

        # Agrupar por ESTU_DEPTO_RESIDE y calcular la media del puntaje seleccionado
        df_agrupado = df_limpio.groupby('ESTU_DEPTO_RESIDE')[puntaje_seleccionado].mean().reset_index()
        mejor_departamento = df_agrupado.loc[df_agrupado[puntaje_seleccionado].idxmax()]
        peor_departamento = df_agrupado.loc[df_agrupado[puntaje_seleccionado].idxmin()]
        
        # Gráfico de barras
        plt.figure(figsize=(14, 8))
        df_comparacion = pd.DataFrame([mejor_departamento, peor_departamento]).sort_values(by=puntaje_seleccionado, ascending=False)
        bar_plot = sns.barplot(data=df_comparacion, y='ESTU_DEPTO_RESIDE', x=puntaje_seleccionado, 
                               palette=['#006400', '#8B0000'])

        plt.title(f'Comparativa del {puntaje_seleccionado.replace("_", " ")}: Mejor vs Peor Departamento', 
                  fontsize=18, weight='bold')
        plt.xlabel(f'Media del {puntaje_seleccionado.replace("_", " ")}', fontsize=16, fontweight='bold')
        plt.ylabel('Departamento', fontsize=16, fontweight='bold')

        for p in bar_plot.patches:
            value = round(p.get_width())
            bar_plot.annotate(f'{value}', (p.get_width() / 2, p.get_y() + p.get_height() / 2.),
                              ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        plt.tight_layout()
        st.pyplot(plt)

        # Gráfico de radar
        df_limpio_normalizado = df_limpio.copy()
        columnas_a_normalizar = ['FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 
                                 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'FAMI_NUMLIBROS']
        
        for columna in columnas_a_normalizar:
            min_val = df_limpio_normalizado[columna].min()
            max_val = df_limpio_normalizado[columna].max()
            df_limpio_normalizado[columna] = (df_limpio_normalizado[columna] - min_val) / (max_val - min_val)

        mejor_data_normalizado = df_limpio_normalizado[df_limpio_normalizado['ESTU_DEPTO_RESIDE'] == mejor_departamento['ESTU_DEPTO_RESIDE']]
        peor_data_normalizado = df_limpio_normalizado[df_limpio_normalizado['ESTU_DEPTO_RESIDE'] == peor_departamento['ESTU_DEPTO_RESIDE']]

        promedios_mejor_normalizados = mejor_data_normalizado[columnas_a_normalizar].mean()
        promedios_peor_normalizados = peor_data_normalizado[columnas_a_normalizar].mean()

        nuevas_etiquetas = [
            'Estrato de Vivienda', 
            'Nivel Educativo del Padre', 
            'Nivel Educativo de la Madre', 
            'Acceso a Internet', 
            'Disponibilidad de Computadora', 
            'Número de Libros del Hogar'
        ]

        promedios_mejor = promedios_mejor_normalizados.tolist() + [promedios_mejor_normalizados.tolist()[0]]
        promedios_peor = promedios_peor_normalizados.tolist() + [promedios_peor_normalizados.tolist()[0]]

        angles = np.linspace(0, 2 * np.pi, len(nuevas_etiquetas), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), dpi=100, subplot_kw=dict(polar=True))

        ax.plot(angles, promedios_mejor, color='green', linewidth=2, linestyle='solid', label=mejor_departamento['ESTU_DEPTO_RESIDE'])
        ax.fill(angles, promedios_mejor, color='green', alpha=0.25)

        ax.plot(angles, promedios_peor, color='red', linewidth=2, linestyle='solid', label=peor_depart
