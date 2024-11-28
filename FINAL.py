import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Dashboard Interactivo", layout="wide")
st.title("Dashboard Interactivo de Análisis de Datos")

# Carga de datos
uploaded_file = st.file_uploader("Sube un archivo .parquet", type="parquet")
if uploaded_file is not None:
    df = pd.read_parquet(uploaded_file)
    st.success("Datos cargados correctamente.")

    # Selección de puntaje para análisis
    puntajes = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    selected_score = st.selectbox("Selecciona un puntaje para comparar departamentos", puntajes)

    # Procesamiento de datos para gráfico de barras
    top_departamento = df.loc[df[selected_score].idxmax()]
    bottom_departamento = df.loc[df[selected_score].idxmin()]
    bar_data = pd.DataFrame({
        "Departamento": [top_departamento["Departamento"], bottom_departamento["Departamento"]],
        "Puntaje": [top_departamento[selected_score], bottom_departamento[selected_score]]
    })

    # Gráfico de barras
    st.subheader("Comparación de Mejor y Peor Departamento")
    bar_fig = px.bar(bar_data, x="Departamento", y="Puntaje", text="Puntaje",
                     title=f"Mejor y Peor Departamento en {selected_score}",
                     labels={"Puntaje": selected_score},
                     color="Departamento")
    bar_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    st.plotly_chart(bar_fig, use_container_width=True)

    # Selección de departamentos para análisis radar
    st.subheader("Comparación de Factores Socioeconómicos")
    departamentos = st.multiselect("Selecciona departamentos para comparar",
                                    options=df["Departamento"].unique(),
                                    default=[top_departamento["Departamento"], bottom_departamento["Departamento"]])

    if departamentos:
        radar_data = df[df["Departamento"].isin(departamentos)].set_index("Departamento")
        socio_factors = st.multiselect("Selecciona factores socioeconómicos para incluir en el gráfico radar",
                                       options=puntajes, default=puntajes[:5])

        if socio_factors:
            radar_fig = go.Figure()
            for dept in departamentos:
                radar_fig.add_trace(go.Scatterpolar(
                    r=radar_data.loc[dept, socio_factors].values,
                    theta=socio_factors,
                    fill='toself',
                    name=dept
                ))

            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, radar_data[socio_factors].max().max()])
                ),
                showlegend=True,
                title="Gráfico Radar de Factores Socioeconómicos"
            )
            st.plotly_chart(radar_fig, use_container_width=True)
else:
    st.info("Por favor, sube un archivo .parquet para comenzar.")
