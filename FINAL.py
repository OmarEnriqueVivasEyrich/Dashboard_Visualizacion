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
    













    
   import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np

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

# Crear gráfico de barras horizontales en una columna
col1, col2 = st.columns(2)

with col1:
    # Crear un gráfico de barras horizontales para el puntaje seleccionado
    plt.figure(figsize=(6, 3))

    # Ordenar los datos de mayor a menor
    df_comparacion = df_comparacion.sort_values(by=puntaje_seleccionado, ascending=False)

    # Crear el gráfico de barras horizontales
    bar_plot = sns.barplot(data=df_comparacion, y='ESTU_DEPTO_RESIDE', x=puntaje_seleccionado, 
                           palette=['#006400', '#8B0000'])

    # Título llamativo con negrita
    plt.title(f'Comparativa del {puntaje_seleccionado.replace("_", " ")}: Mejor vs Peor Departamento', 
              fontsize=10, weight='bold', color='black')

    # Etiquetas de los ejes en negrita y tamaño 10
    plt.xlabel(f'Media del {puntaje_seleccionado.replace("_", " ")}', fontsize=10, fontweight='bold')
    plt.ylabel('Departamento', fontsize=10, fontweight='bold')

    # Añadir los valores redondeados en el centro de las barras, en blanco
    for p in bar_plot.patches:
        value = round(p.get_width())  # Redondear el valor a entero
        bar_plot.annotate(f'{value}', 
                          (p.get_width() / 2, p.get_y() + p.get_height() / 2.),  # Posicionar en el centro de la barra
                          ha='center', va='center', fontsize=10, fontweight='bold', color='white')  # Texto blanco

    # Ajustar el diseño para evitar el recorte de etiquetas
    plt.tight_layout()

    # Mostrar el gráfico
    st.pyplot(plt)

with col2:
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
    
    # Preparar los datos para el radar
    promedios_mejor = promedios_mejor_normalizados.tolist()
    promedios_peor = promedios_peor_normalizados.tolist()
    
    # Número de categorías
    num_vars = len(nuevas_etiquetas)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el gráfico
    
    promedios_mejor += promedios_mejor[:1]
    promedios_peor += promedios_peor[:1]
    
    # Nombres de los departamentos
    mejor_nombre = mejor_departamento['ESTU_DEPTO_RESIDE']
    peor_nombre = peor_departamento['ESTU_DEPTO_RESIDE']
    
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100, subplot_kw=dict(polar=True))
    
    # Crear gráfico de radar para el mejor departamento
    ax.plot(angles, promedios_mejor, color='green', linewidth=2, linestyle='solid', label=mejor_nombre)
    ax.fill(angles, promedios_mejor, color='green', alpha=0.25)
    
    # Crear gráfico de radar para el peor departamento
    ax.plot(angles, promedios_peor, color='red', linewidth=2, linestyle='solid', label=peor_nombre)
    ax.fill(angles, promedios_peor, color='red', alpha=0.25)
    
    # Añadir etiquetas
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(nuevas_etiquetas, fontsize=10, color='black', fontweight='bold')  # Etiquetas con tamaño 10
    
    # Título y leyenda
    ax.set_title('Comparación Normalizada entre Mejor y Peor', fontsize=12, color='black', fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10, frameon=True, shadow=True, fancybox=True)
    
    # Ajustar el diseño
    plt.tight_layout()

    # Mostrar el gráfico
    st.pyplot(fig)

    
    










    
    
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
    
    











    

    # Crear el DataFrame con los puntajes
    df_mapa = df[['ESTU_DEPTO_RESIDE', 'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 
        'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL']]
    
    # Coordenadas de los departamentos
    coordenadas = {
        'ANTIOQUIA': (6.702032125, -75.50455704),
        'ATLANTICO': (10.67700953, -74.96521949),
        'BOGOTÁ': (4.316107698, -74.1810727),
        'BOLIVAR': (8.079796863, -74.23514814),
        'BOYACA': (5.891672889, -72.62788054),
        'CALDAS': (5.280139978, -75.27498304),
        'CAQUETA': (0.798556195, -73.95946756),
        'CAUCA': (2.396833887, -76.82423283),
        'CESAR': (9.53665993, -73.51783154),
        'CORDOBA': (8.358549754, -75.79200872),
        'CUNDINAMARCA': (4.771120716, -74.43111092),
        'CHOCO': (5.397581542, -76.942811),
        'HUILA': (2.570143029, -75.58434836),
        'LA GUAJIRA': (11.47687008, -72.42951072),
        'MAGDALENA': (10.24738355, -74.26175733),
        'META': (3.345562732, -72.95645988),
        'NARIÑO': (1.571094987, -77.87020496),
        'NORTE SANTANDER': (8.09513751, -72.88188297),
        'QUINDIO': (4.455241567, -75.68962853),
        'RISARALDA': (5.240757239, -76.00244469),
        'SANTANDER': (6.693633184, -73.48600894),
        'SUCRE': (9.064941448, -75.10981755),
        'TOLIMA': (4.03477252, -75.2558271),
        'VALLE': (3.569858693, -76.62850427),
        'ARAUCA': (6.569577215, -70.96732394),
        'CASANARE': (5.404064237, -71.60188073),
        'PUTUMAYO': (0.3673031, -75.51406183),
        'SAN ANDRES': (12.54311512, -81.71762382),
        'AMAZONAS': (-1.54622768, -71.50212858),
        'GUAINIA': (2.727842865, -68.81661272),
        'GUAVIARE': (1.924531973, -72.12859569),
        'VAUPES': (0.64624561, -70.56140566),
        'VICHADA': (4.713557125, -69.41400011)
    }
    
    # Excluir departamentos no deseados
    excluir = ['EXTRANJERO', None]
    df_mapa = df_mapa[~df_mapa['ESTU_DEPTO_RESIDE'].isin(excluir)]
    
    # Calcular el promedio de todas las columnas numéricas por departamento
    promedios = df_mapa.groupby('ESTU_DEPTO_RESIDE').mean().reset_index()
    
    # Agregar coordenadas
    promedios['LATITUD'] = promedios['ESTU_DEPTO_RESIDE'].map(lambda x: coordenadas[x][0] if x in coordenadas else None)
    promedios['LONGITUD'] = promedios['ESTU_DEPTO_RESIDE'].map(lambda x: coordenadas[x][1] if x in coordenadas else None)
    















    
    import folium
    import pandas as pd
    from streamlit_folium import st_folium
    
    # Crear un mapa centrado en Colombia con un zoom adecuado
    mapa = folium.Map(location=[4.5709, -74.2973], zoom_start=5, control_scale=True)
    
    # Calcular los límites de los puntajes
    min_puntaje = promedios[puntaje_seleccionado].min()
    max_puntaje = promedios[puntaje_seleccionado].max()
    
    # Definir una función para asignar colores según el puntaje
    def get_color(puntaje):
        rango = max_puntaje - min_puntaje
        if rango == 0:  # Evitar divisiones por cero
            return 'blue'
        if puntaje >= min_puntaje + 0.67 * rango:
            return 'red'  # Alto
        elif puntaje >= min_puntaje + 0.33 * rango:
            return 'orange'  # Medio
        else:
            return 'blue'  # Bajo
    
    # Añadir los puntos de los departamentos con sus puntajes
    for index, row in promedios.iterrows():
        if pd.notnull(row['LATITUD']) and pd.notnull(row['LONGITUD']):  # Asegurarse de que las coordenadas no estén vacías
            # Obtener el color basado en el puntaje
            color = get_color(row[puntaje_seleccionado])
    
            # Crear un marcador con el color basado en el puntaje
            folium.CircleMarker(
                location=[row['LATITUD'], row['LONGITUD']],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(
                    f"<strong>{row['ESTU_DEPTO_RESIDE']}</strong><br>"
                    f"Puntaje promedio: {round(row[puntaje_seleccionado], 0)}",  # Redondeo a entero
                    max_width=300
                ),
            ).add_to(mapa)
    
    # Añadir un título en la parte superior del mapa utilizando un Div
    title_html = '''
        <div style="position: absolute; 
                     top: 10px; left: 50%; 
                     transform: translateX(-50%);
                     font-size: 18px; font-weight: bold; 
                     background-color: rgba(255, 255, 255, 0.7); 
                     padding: 5px 15px; 
                     border-radius: 5px; 
                     z-index: 9999;">
            <h4>Mapa de Puntajes Promedio por Departamento</h4>
        </div>
    '''
    mapa.get_root().html.add_child(folium.Element(title_html))
    
    # Crear una leyenda personalizada
    legend_html = """
    <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 160px; height: 130px; 
                 background-color: white; opacity: 0.9; z-index: 9999; 
                 border: 2px solid grey; border-radius: 5px; padding: 10px; font-size: 12px;">
        <b>Relación Puntaje - Color</b><br>
        <i style="background: red; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></i> Alto<br>
        <i style="background: orange; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></i> Medio<br>
        <i style="background: blue; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></i> Bajo
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(legend_html))
    
    # Integrar el mapa en Streamlit
    st_folium(mapa, width=800, height=500)


    
    
except Exception as e:
    st.error(f"Error al cargar el archivo Parquet: {e}")
