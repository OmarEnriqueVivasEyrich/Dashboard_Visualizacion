import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Asegúrate de cargar el DataFrame df correctamente. Por ejemplo:
# df = pd.read_csv('tu_archivo.csv')

# Verificar si la columna 'PUNT_GLOBAL' existe en el DataFrame
if 'PUNT_GLOBAL' not in df.columns:
    st.error("La columna 'PUNT_GLOBAL' no se encuentra en el DataFrame.")
else:
    # Filtrar el mejor y peor departamento basado en PUNT_GLOBAL
    mejor_departamento = df.loc[df['PUNT_GLOBAL'].idxmax()]
    peor_departamento = df.loc[df['PUNT_GLOBAL'].idxmin()]

    # Asegurarnos de que el DataFrame tiene las columnas necesarias
    columnas_requeridas = ['ESTU_DEPTO_RESIDE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 
                           'FAMI_EDUCACIONMADRE', 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 
                           'FAMI_NUMLIBROS', 'PUNT_GLOBAL']
    
    columnas_no_encontradas = [col for col in columnas_requeridas if col not in df.columns]
    if columnas_no_encontradas:
        st.error(f"Las siguientes columnas no se encuentran en el DataFrame: {', '.join(columnas_no_encontradas)}")
    else:
        # Procesamiento de las columnas
        df_radar = df[columnas_requeridas]

        df_radar['FAMI_ESTRATOVIVIENDA'] = df_radar['FAMI_ESTRATOVIVIENDA'].replace({'Sin Estrato': None}).str.replace('Estrato ', '', regex=False).astype(float)

        # Diccionario de niveles de educación
        orden_educacion = [
            ('Postgrado', 13),
            ('Educación profesional completa', 12),
            ('Educación profesional incompleta', 11),
            ('Técnica o tecnológica completa', 10),
            ('Secundaria (Bachillerato) completa', 9),
            ('Primaria completa', 8),
            ('Técnica o tecnológica incompleta', 7),
            ('Secundaria (Bachillerato) incompleta', 6),
            ('Primaria incompleta', 5),
            ('Ninguno', 4),
            ('No Aplica', 3),
            ('No sabe', 2),
            (None, 1)
        ]
        diccionario_educacion = dict(orden_educacion)

        df_radar['FAMI_EDUCACIONPADRE'] = df_radar['FAMI_EDUCACIONPADRE'].replace(diccionario_educacion)
        df_radar['FAMI_EDUCACIONMADRE'] = df_radar['FAMI_EDUCACIONMADRE'].replace(diccionario_educacion)

        df_radar['FAMI_TIENEINTERNET'] = df_radar['FAMI_TIENEINTERNET'].replace({'Sí': 1, 'No': 0, 'Si': 1}).astype(float)
        df_radar['FAMI_TIENECOMPUTADOR'] = df_radar['FAMI_TIENECOMPUTADOR'].replace({'Sí': 1, 'No': 0, 'Si': 1}).astype(float)

        # Diccionario de niveles de libros
        orden_libros = [
            ('MÁS DE 100 LIBROS', 5),
            ('26 A 100 LIBROS', 4),
            ('11 A 25 LIBROS', 3),
            ('0 A 10 LIBROS', 2),
            (None, 1)
        ]
        diccionario_libros = dict(orden_libros)
        df_radar['FAMI_NUMLIBROS'] = df_radar['FAMI_NUMLIBROS'].replace(diccionario_libros).astype(float)

        # Normalización de las columnas numéricas
        df_radar_normalizado = df_radar.copy()
        columnas_a_normalizar = ['FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 
                                 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'FAMI_NUMLIBROS']

        for columna in columnas_a_normalizar:
            min_val = df_radar_normalizado[columna].min()
            max_val = df_radar_normalizado[columna].max()
            df_radar_normalizado[columna] = (df_radar_normalizado[columna] - min_val) / (max_val - min_val)

        # Filtrar los datos normalizados para el mejor y peor departamento
        mejor_departamento_normalizado = df_radar_normalizado[df_radar_normalizado['ESTU_DEPTO_RESIDE'] == mejor_departamento['ESTU_DEPTO_RESIDE']]
        peor_departamento_normalizado = df_radar_normalizado[df_radar_normalizado['ESTU_DEPTO_RESIDE'] == peor_departamento['ESTU_DEPTO_RESIDE']]

        # Calcular los promedios normalizados
        promedios_mejor_normalizados = mejor_departamento_normalizado[columnas_a_normalizar].mean()
        promedios_peor_normalizados = peor_departamento_normalizado[columnas_a_normalizar].mean()

        # Etiquetas para el gráfico de radar
        nuevas_etiquetas = [
            'Estrato de Vivienda', 
            'Nivel Educativo del Padre', 
            'Nivel Educativo de la Madre', 
            'Acceso a Internet', 
            'Disponibilidad de Computadora', 
            'Número de Libros del Hogar'
        ]

        # Preparar los datos para el radar
        promedios_mejor = promedios_mejor_normalizados.tolist()
        promedios_peor = promedios_peor_normalizados.tolist()

        # Número de categorías
        num_vars = len(nuevas_etiquetas)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el gráfico

        promedios_mejor += promedios_mejor[:1]
        promedios_peor += promedios_peor[:1]

        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(7, 7), dpi=100, subplot_kw=dict(polar=True))

        # Crear gráfico de radar para el mejor departamento
        ax.plot(angles, promedios_mejor, color='green', linewidth=2, linestyle='solid', label=mejor_departamento['ESTU_DEPTO_RESIDE'])
        ax.fill(angles, promedios_mejor, color='green', alpha=0.25)

        # Crear gráfico de radar para el peor departamento
        ax.plot(angles, promedios_peor, color='red', linewidth=2, linestyle='solid', label=peor_departamento['ESTU_DEPTO_RESIDE'])
        ax.fill(angles, promedios_peor, color='red', alpha=0.25)

        # Añadir etiquetas con letras negras y tamaño 10
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(nuevas_etiquetas, fontsize=10, color='black', fontweight='bold')  # Etiquetas con tamaño 10

        # Título y leyenda con tamaño de letra 10
        ax.set_title(f'Comparación Normalizada entre {mejor_departamento["ESTU_DEPTO_RESIDE"]} y {peor_departamento["ESTU_DEPTO_RESIDE"]}', fontsize=12, color='black', fontweight='bold', y=1.1)  # Título con tamaño 10
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10, frameon=True, shadow=True, fancybox=True)  # Leyenda con tamaño 10

        # Ajustar el diseño para evitar recortes
        plt.tight_layout()

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
