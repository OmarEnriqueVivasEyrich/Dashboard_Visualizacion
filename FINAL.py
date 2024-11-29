import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración general del Dashboard
st.set_page_config(page_title="Dashboard de Puntajes y Estratos", layout="wide")
st.title("Dashboard de Puntajes y Estratos por Departamento")

# Definir la ruta del archivo Parquet
file_path = 'parquet.parquet'

# Verificar si el archivo existe y cargar los datos
try:
    df = pd.read_parquet(file_path)

    # Preprocesamiento
    def limpiar_datos(df):
        columnas_interes = [
            'ESTU_DEPTO_RESIDE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE',
            'FAMI_EDUCACIONMADRE', 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
            'FAMI_NUMLIBROS', 'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS',
            'PUNT_C_NATURALES', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL'
        ]
        df = df[columnas_interes]

        # Limpieza de FAMI_ESTRATOVIVIENDA
        df['FAMI_ESTRATOVIVIENDA'] = (
            df['FAMI_ESTRATOVIVIENDA']
            .replace({'Sin Estrato': None})
            .str.replace('Estrato ', '', regex=False)
            .astype(float)
        )

        # Mapeo de niveles educativos
        niveles_educacion = {
            'Postgrado': 13, 'Educación profesional completa': 12, 
            'Educación profesional incompleta': 11, 'Técnica o tecnológica completa': 10,
            'Secundaria (Bachillerato) completa': 9, 'Primaria completa': 8,
            'Técnica o tecnológica incompleta': 7, 'Secundaria (Bachillerato) incompleta': 6,
            'Primaria incompleta': 5, 'Ninguno': 4, 'No Aplica': 3, 'No sabe': 2, None: 1
        }
        df['FAMI_EDUCACIONPADRE'] = df['FAMI_EDUCACIONPADRE'].replace(niveles_educacion)
        df['FAMI_EDUCACIONMADRE'] = df['FAMI_EDUCACIONMADRE'].replace(niveles_educacion)

        # Convertir a numérico otros datos
        df['FAMI_TIENEINTERNET'] = df['FAMI_TIENEINTERNET'].replace({'Sí': 1, 'No': 0}).astype(float)
        df['FAMI_TIENECOMPUTADOR'] = df['FAMI_TIENECOMPUTADOR'].replace({'Sí': 1, 'No': 0}).astype(float)

        # Mapeo de número de libros
        libros_mapeo = {
            'MÁS DE 100 LIBROS': 5, '26 A 100 LIBROS': 4,
            '11 A 25 LIBROS': 3, '0 A 10 LIBROS': 2, None: 1
        }
        df['FAMI_NUMLIBROS'] = df['FAMI_NUMLIBROS'].replace(libros_mapeo).astype(float)

        return df

    df_limpio = limpiar_datos(df)

    # Selección de puntaje
    puntajes = [
        'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES',
        'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL'
    ]
    puntaje_seleccionado = st.sidebar.selectbox("Seleccione el puntaje:", puntajes)

    # Cálculo de estadísticas
    df_agrupado = df_limpio.groupby('ESTU_DEPTO_RESIDE')[puntaje_seleccionado].mean().reset_index()
    mejor_departamento = df_agrupado.loc[df_agrupado[puntaje_seleccionado].idxmax()]
    peor_departamento = df_agrupado.loc[df_agrupado[puntaje_seleccionado].idxmin()]

    st.write(f"**Mejor departamento:** {mejor_departamento['ESTU_DEPTO_RESIDE']} - {mejor_departamento[puntaje_seleccionado]:.2f}")
    st.write(f"**Peor departamento:** {peor_departamento['ESTU_DEPTO_RESIDE']} - {peor_departamento[puntaje_seleccionado]:.2f}")

    # Gráfico de barras (Comparativa)
    st.subheader("Comparativa de Mejor y Peor Departamento")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=pd.DataFrame([mejor_departamento, peor_departamento]),
        x=puntaje_seleccionado, y="ESTU_DEPTO_RESIDE",
        palette=['#2ecc71', '#e74c3c'], ax=ax
    )
    ax.set_title(f"Comparativa del Puntaje: {puntaje_seleccionado}")
    ax.set_xlabel("Promedio del Puntaje")
    ax.set_ylabel("Departamento")
    st.pyplot(fig)

    # Radar chart (Normalización)
    st.subheader("Gráfico de Radar")
    columnas_normalizar = [
        'FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE',
        'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'FAMI_NUMLIBROS'
    ]
    normalizados = df_limpio[columnas_normalizar].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
    )
    mejor_norm = normalizados.loc[df_limpio['ESTU_DEPTO_RESIDE'] == mejor_departamento['ESTU_DEPTO_RESIDE']].mean()
    peor_norm = normalizados.loc[df_limpio['ESTU_DEPTO_RESIDE'] == peor_departamento['ESTU_DEPTO_RESIDE']].mean()

    categorias = columnas_normalizar
    angles = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
    angles += angles[:1]
    mejor_data = mejor_norm.tolist() + mejor_norm.tolist()[:1]
    peor_data = peor_norm.tolist() + peor_norm.tolist()[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, mejor_data, color="green", alpha=0.25)
    ax.fill(angles, peor_data, color="red", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorias)
    ax.legend(["Mejor Departamento", "Peor Departamento"])
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error al cargar los datos: {str(e)}")
