import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
import numpy as np
import folium

# Definir la ruta del archivo Parquet
file_path = 'parquet.parquet'  # Cambiado a ruta relativa

# Configuración de estilo
st.set_page_config(page_title="Dashboard de Puntajes y Estratos", layout="wide")
st.title('Dashboard de Puntajes y Estratos por Departamento')

# Verificar si el archivo Parquet existe
# Cargar el archivo Parquet
df = pd.read_parquet(file_path)

try:
    # Realizar el procesamiento previo (asegurándonos de limpiar correctamente los datos)
    df_limpio = df[['ESTU_DEPTO_RESIDE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'FAMI_NUMLIBROS', 'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 
                   'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL']]

    # Reemplazar y convertir FAMI_ESTRATOVIVIENDA
    df_limpio['FAMI_ESTRATOVIVIENDA'] = df_limpio['FAMI_ESTRATOVIVIENDA'].replace({'Sin Estrato': None}).str.replace('Estrato ', '', regex=False).astype(float)

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

    # Reemplazar educación
    df_limpio['FAMI_EDUCACIONPADRE'] = df_limpio['FAMI_EDUCACIONPADRE'].replace(diccionario_educacion)
    df_limpio['FAMI_EDUCACIONMADRE'] = df_limpio['FAMI_EDUCACIONMADRE'].replace(diccionario_educacion)

    # Convertir a numérico FAMI_TIENEINTERNET y FAMI_TIENECOMPUTADOR
    df_limpio['FAMI_TIENEINTERNET'] = df_limpio['FAMI_TIENEINTERNET'].replace({'Sí': 1, 'No': 0, 'Si': 1}).astype(float)
    df_limpio['FAMI_TIENECOMPUTADOR'] = df_limpio['FAMI_TIENECOMPUTADOR'].replace({'Sí': 1, 'No': 0, 'Si': 1}).astype(float)

    # Diccionario de niveles de libros
    orden_libros = [
        ('MÁS DE 100 LIBROS', 5),
        ('26 A 100 LIBROS', 4),
        ('11 A 25 LIBROS', 3),
        ('0 A 10 LIBROS', 2),
        (None, 1)
    ]
    diccionario_libros = dict(orden_libros)

    # Reemplazar libros
    df_limpio['FAMI_NUMLIBROS'] = df_limpio['FAMI_NUMLIBROS'].replace(diccionario_libros).astype(float)

    # Lista de columnas de puntajes disponibles
    puntaje_opciones = [
        'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 
        'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL'
    ]

    # Preguntar al usuario qué puntaje desea utilizar
    puntaje_seleccionado = st.selectbox(
        "Seleccione el puntaje para realizar el gráfico:",
        puntaje_opciones
    )
    
    st.write(f"Ha seleccionado: {puntaje_seleccionado}")

    # Agrupar por ESTU_DEPTO_RESIDE y calcular la media del puntaje seleccionado
    df_agrupado = df_limpio.groupby('ESTU_DEPTO_RESIDE')[puntaje_seleccionado].mean().reset_index()

    # Identificar el mejor y peor departamento según el puntaje seleccionado
    mejor_departamento = df_agrupado.loc[df_agrupado[puntaje_seleccionado].idxmax()]
    peor_departamento = df_agrupado.loc[df_agrupado[puntaje_seleccionado].idxmin()]

    # Crear un DataFrame con solo el mejor y peor departamento
    df_comparacion = pd.DataFrame([mejor_departamento, peor_departamento])

    # Configurar el estilo de seaborn
    sns.set(style="whitegrid")

    # Crear un gráfico de barras horizontales para el puntaje seleccionado
    plt.figure(figsize=(14, 8))

    # Ordenar los datos de mayor a menor
    df_comparacion = df_comparacion.sort_values(by=puntaje_seleccionado, ascending=False)

    # Crear el gráfico de barras horizontales
    bar_plot = sns.barplot(data=df_comparacion, y='ESTU_DEPTO_RESIDE', x=puntaje_seleccionado, 
                           palette=['#006400', '#8B0000'])

    # Título llamativo con negrita
    plt.title(f'Comparativa del {puntaje_seleccionado.replace("_", " ")}: Mejor vs Peor Departamento', 
              fontsize=18, weight='bold', color='black')

    # Etiquetas de los ejes en negrita y tamaño 16
    plt.xlabel(f'Media del {puntaje_seleccionado.replace("_", " ")}', fontsize=16, fontweight='bold')
    plt.ylabel('Departamento', fontsize=16, fontweight='bold')

    # Cambiar tamaño de fuente de los nombres de los departamentos y ponerlos en negrita
    bar_plot.set_yticklabels(bar_plot.get_yticklabels(), fontsize=16, fontweight='bold', color='black')

    # Añadir los valores redondeados en el centro de las barras, en blanco
    for p in bar_plot.patches:
        value = round(p.get_width())  # Redondear el valor a entero
        bar_plot.annotate(f'{value}', 
                          (p.get_width() / 2, p.get_y() + p.get_height() / 2.),  # Posicionar en el centro de la barra
                          ha='center', va='center', fontsize=16, fontweight='bold', color='white')  # Texto blanco

    # Fondo blanco para la figura y los ejes
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    plt.gca().set_facecolor('white')

    # Hacer los números del eje X de tamaño 16
    plt.tick_params(axis='x', labelsize=16)

    # Ajustar el diseño para evitar el recorte de etiquetas
    plt.tight_layout()

    # Mostrar el gráfico
    st.pyplot(plt)

    # Supongamos que tienes el DataFrame original df_radar ya procesado
    # Normalizar las columnas numéricas usando Min-Max
    df_limpio_normalizado = df_limpio.copy()
    columnas_a_normalizar = ['FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 
                             'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'FAMI_NUMLIBROS']
    
    for columna in columnas_a_normalizar:
        min_val = df_limpio_normalizado[columna].min()
        max_val = df_limpio_normalizado[columna].max()
        df_limpio_normalizado[columna] = (df_limpio_normalizado[columna] - min_val) / (max_val - min_val)
    
    # Filtrar los datos normalizados para Bogotá y Chocó
    mejor_data_normalizado = df_limpio_normalizado[df_limpio_normalizado['ESTU_DEPTO_RESIDE'] == mejor_departamento['ESTU_DEPTO_RESIDE']]
    peor_data_normalizado = df_limpio_normalizado[df_limpio_normalizado['ESTU_DEPTO_RESIDE'] == peor_departamento['ESTU_DEPTO_RESIDE']]
    
    # Calcular los promedios normalizados
    promedios_mejor_normalizados = mejor_data_normalizado[columnas_a_normalizar].mean()
    promedios_peor_normalizados = peor_data_normalizado[columnas_a_normalizar].mean()
    
    # Mejorar los nombres de las etiquetas para que sean más descriptivos
    nuevas_etiquetas = [
        'Estrato de Vivienda', 
        'Nivel Educativo del Padre', 
        'Nivel Educativo de la Madre', 
        'Acceso a Internet', 
        'Acceso a Computadora', 
        'Número de Libros en Casa'
    ]
    
    # Generar gráfico de radar con los promedios normalizados
    categories = nuevas_etiquetas
    values_mejor = promedios_mejor_normalizados
    values_peor = promedios_peor_normalizados

    # Gráfico de Radar
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Número de categorías
    num_vars = len(categories)

    # Ángulos para las categorías
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Hacer que el gráfico sea circular cerrando el ciclo
    values_mejor = values_mejor.tolist()
    values_mejor += values_mejor[:1]
    values_peor = values_peor.tolist()
    values_peor += values_peor[:1]
    angles += angles[:1]

    # Graficar las líneas de las dos series
    ax.plot(angles, values_mejor, linewidth=2, linestyle='solid', label='Mejor Departamento', color='green')
    ax.plot(angles, values_peor, linewidth=2, linestyle='solid', label='Peor Departamento', color='red')

    # Rellenar las áreas de las líneas
    ax.fill(angles, values_mejor, color='green', alpha=0.25)
    ax.fill(angles, values_peor, color='red', alpha=0.25)

    # Añadir etiquetas
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

    # Título
    ax.set_title('Comparación Normalizada: Mejor vs Peor Departamento', size=16, color='black', fontweight='bold')

    # Leyenda
    ax.legend(loc='upper right', fontsize=12, frameon=False)

    # Mostrar el gráfico
    st.pyplot(fig)

except Exception as e:
    st.write(f"Error al procesar el archivo Parquet: {e}")
