import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
import pandas as pd

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
        'Disponibilidad de Computadora', 
        'Número de Libros del Hogar'
    ]
    
    # Crear gráfico de radar
    # Preparar los datos para el radar
    promedios_mejor = promedios_mejor_normalizados.tolist()
    promedios_peor = promedios_peor_normalizados.tolist()
    
    # Número de categorías
    num_vars = len(nuevas_etiquetas)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el gráfico
    
    promedios_mejor += promedios_mejor[:1]
    promedios_peor += promedios_peor[:1]
    
    # Nombres departamntos
    
    mejor_nombre=mejor_departamento['ESTU_DEPTO_RESIDE']
    peor_nombre=peor_departamento['ESTU_DEPTO_RESIDE']
    
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100, subplot_kw=dict(polar=True))
    
    # Crear gráfico de radar para Bogotá
    ax.plot(angles, promedios_mejor, color='green', linewidth=2, linestyle='solid', label=mejor_nombre)
    ax.fill(angles, promedios_mejor, color='green', alpha=0.25)
    
    # Crear gráfico de radar para Chocó
    ax.plot(angles, promedios_peor, color='red', linewidth=2, linestyle='solid', label=peor_nombre)
    ax.fill(angles, promedios_peor, color='red', alpha=0.25)
    
    # Añadir etiquetas con letras negras y tamaño 10
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(nuevas_etiquetas, fontsize=10, color='black', fontweight='bold')  # Etiquetas con tamaño 10
    
    # Título y leyenda con tamaño de letra 10
    ax.set_title('Comparación Normalizada entre Mejor y Peor', fontsize=12, color='black', fontweight='bold', y=1.1)  # Título con tamaño 10
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10, frameon=True, shadow=True, fancybox=True)  # Leyenda con tamaño 10
    
    # Ajustar el diseño para evitar recortes
    plt.tight_layout()
    # Mostrar el gráfico
    st.pyplot(plt)

    # Actualizar el diccionario de regiones combinando Insular con Caribe y excluyendo EXTRANJERO
    regiones = {
        'CUNDINAMARCA': 'Andina', 'ANTIOQUIA': 'Andina', 'BOLIVAR': 'Caribe', 'NORTE SANTANDER': 'Andina', 'CAQUETA': 'Amazonia',
        'RISARALDA': 'Andina', 'SANTANDER': 'Andina', 'MAGDALENA': 'Caribe', 'BOGOTÁ': 'Andina', 'CALDAS': 'Andina', 
        'VALLE': 'Pacífica', 'QUINDIO': 'Andina', 'SUCRE': 'Caribe', 'ATLANTICO': 'Caribe', 'HUILA': 'Andina', 
        'LA GUAJIRA': 'Caribe', 'CORDOBA': 'Caribe', 'NARIÑO': 'Pacífica', 'META': 'Orinoquía', 'CAUCA': 'Pacífica',
        'CASANARE': 'Orinoquía', 'GUAVIARE': 'Orinoquía', 'PUTUMAYO': 'Amazonia', 'CESAR': 'Caribe', 'CHOCO': 'Pacífica',
        'ARAUCA': 'Orinoquía', 'VICHADA': 'Orinoquía', 'TOLIMA': 'Andina', 'BOYACA': 'Andina', 'AMAZONAS': 'Amazonia',
        'SAN ANDRES': 'Caribe', 'VAUPES': 'Amazonia', 'GUAINIA': 'Amazonia', None: 'Desconocida'
    }
    
    # Calcular las medias por departamento
    df_means = df_limpio.groupby('ESTU_DEPTO_RESIDE').agg(
        media_puntaje=(puntaje_seleccionado, 'mean'),
        media_estrato=('FAMI_ESTRATOVIVIENDA', 'mean')
    ).reset_index()
    
    # Asignar las regiones a cada departamento
    df_means['Region'] = df_means['ESTU_DEPTO_RESIDE'].map(regiones)
    
    # Filtrar para excluir el departamento "EXTRANJERO"
    df_means = df_means[df_means['ESTU_DEPTO_RESIDE'] != 'EXTRANJERO']
    
    # Crear una paleta de colores únicos para cada región
    colores = sns.color_palette("Set2", len(df_means['Region'].unique()))  # Usamos una paleta con suficientes colores únicos
    
    # Crear un gráfico de dispersión para todos los puntos
    plt.figure(figsize=(16, 10))  # Tamaño de la figura
    
    # Configurar el color de fondo de la figura y los ejes a blanco
    fig = plt.gcf()
    fig.patch.set_alpha(1)  # Fondo de la figura opaco (blanco)
    ax = plt.gca()
    ax.set_facecolor('white')  # Fondo blanco de los ejes
    
    # Usamos seaborn para un gráfico más atractivo, asignando colores por región
    for idx, region in enumerate(df_means['Region'].unique()):
        region_data = df_means[df_means['Region'] == region]
        
        # Graficar los puntos de cada región con un color distinto
        sns.scatterplot(data=region_data, x='media_estrato', y='media_puntaje', 
                        color=colores[idx], label=region, s=120)
        
        # Agregar los nombres de los departamentos al lado de cada punto, moviéndolos más arriba
        for i, row in region_data.iterrows():
            plt.text(row['media_estrato'] - 0.05, row['media_puntaje'] + 0.15, row['ESTU_DEPTO_RESIDE'], 
                     fontsize=12, color=colores[idx], alpha=0.7, fontweight='bold')  # Aumentar el tamaño de las letras
    
    # Agregar una única línea de regresión lineal para todos los puntos
    sns.regplot(data=df_means, x='media_estrato', y='media_puntaje', scatter=False, color='black', 
                line_kws={"linewidth": 3, "linestyle": "--"}, ci=None)
    
    # Ajustar el título y las etiquetas con letras más grandes y negritas
    plt.title('Gráfico de Media de Estrato vs Media de Puntaje por Departamento con Línea de Regresión General', 
              fontsize=20, fontweight='bold')  # Ajustar tamaño del título
    plt.xlabel('Media de Estrato', fontsize=16, fontweight='bold')
    plt.ylabel('Media de Puntaje', fontsize=16, fontweight='bold')
    
    # Mostrar la leyenda con letras más grandes y en negrita (pasando fontweight dentro de prop)
    plt.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, 
               title_fontsize=16, prop={'weight': 'bold'}, labels=[region.upper() for region in df_means['Region'].unique()])
    
    # Aumentar el tamaño del cuadro de la leyenda
    plt.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, 
               title_fontsize=16, prop={'weight': 'bold'}, labels=[region.upper() for region in df_means['Region'].unique()],
               markerscale=2, borderpad=2, labelspacing=1.5)
    
    # Configurar el color de las cuadrículas
    ax.grid(True, color='blue', linestyle='--', linewidth=1)  # Establecer el color azul, líneas discontinuas y grosor 1
    
    # Aumentar el tamaño de los números de los ejes X y Y
    plt.tick_params(axis='x', labelsize=16)  # Cambiar tamaño de los números del eje X
    plt.tick_params(axis='y', labelsize=16)  # Cambiar tamaño de los números del eje Y
    
    # Guardar el gráfico con fondo blanco
    plt.tight_layout()
    st.pyplot(plt)
  


    
except Exception as e:
    st.error(f"Error al cargar el archivo Parquet: {e}")
