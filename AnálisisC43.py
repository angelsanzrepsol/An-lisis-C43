
import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import json
from io import BytesIO
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# CONFIGURACIÓN GENERAL
st.set_page_config(
    page_title="Análisis dureza de coque",
    layout="wide"
)

# ESTÉTICA (INDUSTRIAL – COPIADA DE TUS APPS)
st.markdown("""
<style>

/* Fondo general */
html, body, .block-container, [class*="stApp"] {
    background-color: #FFFFFF !important;
    color: #333333 !important;
}

/* Títulos */
h1, h2, h3, h4, h5, h6 {
    color: #D98B3B !important;
    font-weight: 800 !important;
}

/* Título azul oscuro */
.darkblue-title {
    color: #0B1A33 !important;
    font-weight: 800 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] p {
    color: #666666 !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] p {
    color: red !important;
    font-weight: 700 !important;
}

/* Botones */
.stButton>button {
    background-color: #D98B3B !important;
    color: white !important;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# CABECERA + LOGO
st.markdown(
    "<h1 class='darkblue-title'>Análisis dureza del coque</h1>",
    unsafe_allow_html=True
)

logo_path = Path("logo_repsol.png")
if logo_path.exists():
    try:
        logo = Image.open(logo_path).convert("RGBA")
        blur = 12
        pad = blur * 4

        canvas = Image.new(
            "RGBA",
            (logo.width + pad, logo.height + pad),
            (255, 255, 255, 0)
        )

        canvas.paste(logo, (pad // 2, pad // 2), logo)

        mask = canvas.split()[3]
        halo = mask.filter(ImageFilter.GaussianBlur(blur))
        canvas.putalpha(halo)

        st.image(canvas, width=180)
    except Exception:
        st.warning("No se pudo cargar el logo.")
else:
    st.info("Archivo logo_repsol.png no encontrado.")

# FUNCIÓN DE LECTURA DE DATOS DE PROCESO

def leer_datos_csv(uploaded_file):
    try:
        try:
            df = pd.read_csv(uploaded_file, sep=",")
            if df.shape[1] == 1:
                raise ValueError
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";")
        return df
    except Exception:
        return None



def aplicar_exclusion_variables(df, camara):
    excluidas = st.session_state.variables_excluidas_global.get(camara, [])
    return df.drop(columns=[c for c in excluidas if c in df.columns])

def aplicar_filtros_activos(df, camara):
    """
    Aplica los rangos de los filtros activos al dataframe de una cámara.
    """

    # Si no hay filtros activos, devuelve el dataframe tal cual
    if "filtros_activos" not in st.session_state:
        return df

    if not st.session_state.filtros_activos:
        return df

    # Recorrer todos los filtros activos
    for nombre in st.session_state.filtros_activos:

        # Obtener el filtro guardado
        f = st.session_state.filtros_guardados.get(nombre, {})

        # Comprobar que el filtro pertenece a esta cámara
        if f.get("camara") != camara:
            continue

        # Aplicar rangos del filtro
        rangos = f.get("rangos", {})

        for variable, (vmin, vmax) in rangos.items():
            if variable in df.columns:
                df = df[(df[variable] >= vmin) & (df[variable] <= vmax)]

    return df
# SIDEBAR — CARGA DE DATOS DE PROCESO (VARIOS EXCEL = VARIAS CÁMARAS)
st.sidebar.header("Datos de proceso")

uploaded_file = st.sidebar.file_uploader(
    "Subir archivo Excel de proceso",
    type=["xlsx", "xls"]
)


# Inicializar estados UNA SOLA VEZ
if "df_camaras_original" not in st.session_state:
    st.session_state.df_camaras_original = {}

if "df_camaras_activo" not in st.session_state:
    st.session_state.df_camaras_activo = {}

if "df_camaras_eliminados" not in st.session_state:
    st.session_state.df_camaras_eliminados = {}
if "variables_excluidas_global" not in st.session_state:
    st.session_state.variables_excluidas_global = {}

# ESTADOS PARA FILTROS GUARDADOS
if "filtros_guardados" not in st.session_state:
    st.session_state.filtros_guardados = {}

if "filtros_activos" not in st.session_state:
    st.session_state.filtros_activos = set()


# ESTADO PARA DESCARGA TAB 2
if "datos_descarga_tab2" not in st.session_state:
    st.session_state.datos_descarga_tab2 = {}

if uploaded_file:

    try:

        xls = pd.ExcelFile(uploaded_file)
        # obtener fechas en marcha desde hoja General
        df_general = pd.read_excel(
            xls,
            sheet_name="General",
            header=None
        )
        
        fechas_marcha = df_general[
            df_general.iloc[:,1] == "MARCHA"
        ].iloc[:,0]
        
        fechas_marcha = pd.to_datetime(fechas_marcha)

        for hoja in xls.sheet_names:
        
            if hoja == "General":
                continue

            df_raw = pd.read_excel(
                xls,
                sheet_name=hoja,
                header=None
            )
        
            if df_raw is None or df_raw.empty:
                continue
        
            # filas de nombres
            fila_tag = df_raw.iloc[2]
            fila_var = df_raw.iloc[3]
            fila_uni = df_raw.iloc[4]
        
            columnas = []
        
            for i,(tag, var, uni) in enumerate(zip(fila_tag, fila_var, fila_uni)):

                partes = []
            
                if pd.notna(tag):
                    partes.append(str(tag).strip())
            
                if pd.notna(var):
                    partes.append(str(var).strip())
            
                if pd.notna(uni):
                    partes.append(f"({str(uni).strip()})")
            
                nombre = " - ".join(partes)
            
                # si la columna queda vacía le ponemos nombre automático
                if nombre == "":
                    nombre = f"var_{i}"
            
                columnas.append(nombre)
        
            # datos empiezan en fila 5
            df = df_raw.iloc[5:].copy()
        
            df.columns = columnas
            # convertir fecha
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors="coerce")
            
            # convertir fecha
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors="coerce")
            
            # quedarnos solo con fechas en marcha
            df = df[df.iloc[:,0].isin(fechas_marcha)]
        
            # eliminar columnas vacías
            df = df.dropna(axis=1, how="all")
            # convertir columnas a numéricas
            df.iloc[:,2:] = df.iloc[:,2:].apply(pd.to_numeric, errors="coerce")
        
            camara = hoja
        
            st.session_state.df_camaras_original[camara] = df.copy()
            st.session_state.df_camaras_activo[camara] = df.copy()
        
            st.session_state.df_camaras_eliminados[camara] = pd.DataFrame(
                columns=df.columns
            )
        
            st.session_state.variables_excluidas_global[camara] = []

            st.session_state.df_camaras_activo[camara] = df.copy()

            st.session_state.df_camaras_eliminados[camara] = pd.DataFrame(
                columns=df.columns
            )

            st.session_state.variables_excluidas_global[camara] = []

        st.sidebar.success(
            f"Archivo cargado correctamente ({len(xls.sheet_names)} hojas detectadas)"
        )

    except Exception as e:

        st.sidebar.error(
            f"Error leyendo Excel: {e}"
        )

if not st.session_state.df_camaras_original:
    st.sidebar.info("No hay cámaras cargadas todavía")
else:
    st.sidebar.markdown("### Cámaras cargadas")
    for cam in st.session_state.df_camaras_original:
        st.sidebar.write(f"- {cam}")


# PESTAÑAS 
tab1, tab_filtros, tab2, tab3, tab4, tab5 = st.tabs([
    "Visión General",
    "Filtros guardados",
    "Graficado",
    "Correlaciones",
    "Modelo Predictivo",
    "Simulador de Operación"
])

# PESTAÑA 1 
with tab1:
    st.subheader("Visión General — creación de filtros")

    if not st.session_state.df_camaras_original:
        st.info("Cargue datos primero")
        st.stop()

    camara = st.selectbox(
        "Cámara",
        st.session_state.df_camaras_original.keys()
    )
    st.markdown("### Variables a excluir completamente")

    df_original = st.session_state.df_camaras_original[camara]
    
    cols_num_original = df_original.select_dtypes(include="number").columns.tolist()
    
    vars_excluir = st.multiselect(
        "Seleccionar variables a excluir del filtro",
        cols_num_original,
        default=st.session_state.variables_excluidas_global.get(camara, []),
        key=f"vg_vars_excluir_{camara}"
    )
    
    st.session_state.variables_excluidas_global[camara] = vars_excluir

    df = st.session_state.df_camaras_activo[camara]

    df = aplicar_exclusion_variables(df, camara)
    
    df = aplicar_filtros_activos(df, camara)
    cols = df.select_dtypes(include="number").columns.tolist()

    x_var = st.selectbox("Variable base (X)", cols)

    filtros_temp = {}

    st.markdown("### Ajuste de rangos por variable")

    for y in cols:
        if y == x_var:
            continue

        ymin, ymax = float(df[y].min()), float(df[y].max())
        rmin, rmax = st.slider(
            f"{y}",
            ymin, ymax,
            (ymin, ymax),
            key=f"vg_{camara}_{y}"
        )

        filtros_temp[y] = (rmin, rmax)

        df_plot = df[(df[y] >= rmin) & (df[y] <= rmax)]
        fig = px.scatter(df_plot, x=x_var, y=y)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    nombre_filtro = st.text_input("Nombre del filtro")

    if st.button("Guardar filtro"):
        st.session_state.filtros_guardados[nombre_filtro] = {
            "camara": camara,
            "x_var": x_var,
            "rangos": filtros_temp,
            "variables_excluidas": st.session_state.variables_excluidas_global.get(camara, [])
        }

        st.success(f"Filtro '{nombre_filtro}' guardado")
# PESTAÑA — FILTROS GUARDADOS
with tab_filtros:


    st.subheader("Filtros guardados")

    # DESCARGAR TODOS LOS FILTROS
    if st.session_state.filtros_guardados:
        filtros_json = json.dumps(
            st.session_state.filtros_guardados,
            indent=4
        )

        st.download_button(
            "📥 Descargar TODOS los filtros",
            data=filtros_json,
            file_name="filtros_coque.json",
            mime="application/json"
        )

    # IMPORTAR FILTROS
    st.markdown("---")
    st.markdown("### Importar filtros")

    filtro_file = st.file_uploader(
        "Subir archivo de filtros (.json)",
        type=["json"],
        key="upload_filtros"
    )

    if filtro_file is not None:
        try:
            filtros_importados = json.load(filtro_file)

            if isinstance(filtros_importados, dict):
                for nombre, filtro in filtros_importados.items():
                    st.session_state.filtros_guardados[nombre] = filtro

                st.success("Filtros importados correctamente")
            else:
                st.error("El archivo no tiene el formato correcto")

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

    # LISTADO DE FILTROS
    st.markdown("---")
    st.markdown("### Filtros disponibles")

    if not st.session_state.filtros_guardados:
        st.info("No hay filtros guardados")
    else:
        for nombre, f in st.session_state.filtros_guardados.items():
            with st.expander(nombre):
    
                st.write(f"**Cámara:** {f.get('camara', '-')}")
                st.write(f"**Variable X:** {f.get('x_var', '-')}")
                st.markdown("**Rangos:**")
                st.json(f.get("rangos", {}))
    
                st.markdown("### Editar variables excluidas en este filtro")
    
                camara_filtro = f.get("camara")

                # Verificar que la cámara existe
                if camara_filtro not in st.session_state.df_camaras_original:
                
                    st.warning(f"La cámara {camara_filtro} no está cargada actualmente")
                    continue
                
                df_original = st.session_state.df_camaras_original.get(camara_filtro)
                if df_original is None:
                    st.warning(f"La cámara {camara_filtro} no está cargada actualmente")
                    continue


                cols_num = df_original.select_dtypes(include="number").columns.tolist()
    
                vars_actuales = f.get("variables_excluidas", [])
    
                vars_editadas = st.multiselect(
                    "Variables excluidas en este filtro",
                    cols_num,
                    default=vars_actuales,
                    key=f"edit_vars_{nombre}"
                )
    
                # BOTÓN QUE TE FALTA
                if st.button("Guardar cambios", key=f"save_vars_{nombre}"):
    
                    st.session_state.filtros_guardados[nombre]["variables_excluidas"] = vars_editadas
    
                    st.success("Filtro actualizado correctamente")


                col1, col2, col3 = st.columns(3)

                # ACTIVAR FILTRO (SOLO 1 POR CÁMARA)
                camara_filtro = f["camara"]
                activo = nombre in st.session_state.filtros_activos
                
                if col1.checkbox(
                    "Activo",
                    value=activo,
                    key=f"chk_{nombre}"
                ):
                    # Desactivar otros filtros de la MISMA cámara
                    for n in list(st.session_state.filtros_activos):
                        if st.session_state.filtros_guardados[n]["camara"] == camara_filtro:
                            st.session_state.filtros_activos.discard(n)
                
                    st.session_state.filtros_activos.add(nombre)
                    vars_excluidas = f.get("variables_excluidas", [])

                    df_actual = st.session_state.df_camaras_original.get(camara_filtro)
                    
                    if df_actual is not None:
                        vars_excluidas = [v for v in vars_excluidas if v in df_actual.columns]
                    
                    st.session_state.variables_excluidas_global[camara_filtro] = vars_excluidas
                    

                else:
                    st.session_state.filtros_activos.discard(nombre)
                    st.session_state.variables_excluidas_global[camara_filtro] = []
                    
                # ELIMINAR FILTRO
                if col2.button("Eliminar", key=f"eliminar_{nombre}"):
                    del st.session_state.filtros_guardados[nombre]
                    if st.session_state.filtro_activo == nombre:
                        st.session_state.filtro_activo = None
                    st.rerun()

                # DESCARGAR FILTRO INDIVIDUAL
                filtro_individual = {nombre: f}
                filtro_json = json.dumps(filtro_individual, indent=4)

                nombre_archivo = re.sub(
                    r"[^a-zA-Z0-9_-]",
                    "_",
                    nombre
                )

                col3.download_button(
                    "📥 Descargar",
                    data=filtro_json,
                    file_name=f"{nombre_archivo}.json",
                    mime="application/json",
                    key=f"download_{nombre}"
                )

# PESTAÑA 3 — GRAFICADO
with tab2:

    st.subheader("Graficado interactivo avanzado de variables de proceso")

    # COMPROBACIÓN DE DATOS
    if "df_camaras_activo" not in st.session_state:
        st.warning("Cargue primero datos de cámaras.")
        st.stop()

    df_camaras_activo = st.session_state.df_camaras_activo
    df_camaras_eliminados = st.session_state.df_camaras_eliminados
    df_camaras_original = st.session_state.df_camaras_original

    camaras_disponibles = sorted(df_camaras_activo.keys())

    # SELECCIÓN DE CÁMARAS A REPRESENTAR
    camaras_sel = st.multiselect(
        "Cámaras a representar",
        camaras_disponibles,
        default=camaras_disponibles[:1]
    )

    if not camaras_sel:
        st.warning("Seleccione al menos una cámara.")
        st.stop()

    # SELECCIÓN DE CÁMARA FUENTE DE X
    camara_x = st.selectbox(
        "Cámara de referencia para el eje X",
        camaras_sel
    )

    df_x_ref = aplicar_exclusion_variables(
        df_camaras_activo[camara_x],
        camara_x
    )
    
    df_x_ref = aplicar_filtros_activos(df_x_ref, camara_x)
    cols_num_x = df_x_ref.select_dtypes(include="number").columns.tolist()

    if len(cols_num_x) < 1:
        st.error("La cámara de referencia no tiene columnas numéricas.")
        st.stop()

    x_var = st.selectbox(
        "Variable eje X (común)",
        cols_num_x
    )
    # FILTRO PREFIJADO
    filtro_sel = st.selectbox(
        "Filtro prefijado",
        ["(ninguno)"] + list(st.session_state.filtros_guardados.keys())
    )
    
    if filtro_sel != "(ninguno)":

        filtro = st.session_state.filtros_guardados[filtro_sel]
        camara_filtro = filtro["camara"]
    
        # Limpiar filtros activos de esa cámara
        for n in list(st.session_state.filtros_activos):
            if st.session_state.filtros_guardados[n]["camara"] == camara_filtro:
                st.session_state.filtros_activos.discard(n)
    
        # Activar el nuevo
        st.session_state.filtros_activos.add(filtro_sel)
    
        # Aplicar variables excluidas
        st.session_state.variables_excluidas_global[camara_filtro] = filtro.get(
            "variables_excluidas",
            []
        )
    
    else:
        st.session_state.filtros_activos.clear()



    # SELECCIÓN DE Y POR CÁMARA
    st.markdown("### Selección de variables Y por cámara")
    # VARIABLE PARA GRADIENTE DE COLOR
    st.markdown("### Variable para colorear los puntos")
    
    vars_color_posibles = cols_num_x.copy()
    vars_color_posibles.insert(0, "(ninguna)")
    
    color_var = st.selectbox(
        "Variable para gradiente de color",
        vars_color_posibles,
        key="graf_color_var"
    )
    
    if color_var == "(ninguna)":
        color_var = None
    
    y_vars_por_camara = {}

    for camara in camaras_sel:
        df_cam = aplicar_exclusion_variables(
            df_camaras_activo[camara],
            camara
        )
    
        df_cam = aplicar_filtros_activos(df_cam, camara)

        cols_cam = df_cam.select_dtypes(include="number").columns.tolist()

        y_sel = st.multiselect(
            f"Variables Y para cámara {camara}",
            [c for c in cols_cam if c != x_var],
            default=[c for c in cols_cam if c != x_var][:1],
            key=f"y_sel_{camara}"
        )

        if y_sel:
            y_vars_por_camara[camara] = y_sel

    if not y_vars_por_camara:
        st.warning("Seleccione al menos una variable Y.")
        st.stop()

    # BOTÓN RESTAURAR
    if st.button("Restaurar todo"):
        st.session_state.df_camaras_activo = {
            k: v.copy() for k, v in df_camaras_original.items()
        }
        st.session_state.df_camaras_eliminados = {
            k: pd.DataFrame(columns=v.columns)
            for k, v in df_camaras_original.items()
        }
        st.session_state.axis_frozen_tab2 = False
        st.session_state.axis_limits_tab2 = {}
        st.rerun()

    # FILTRO POR X (USANDO CÁMARA DE REFERENCIA)
    xmin = float(df_x_ref[x_var].min())
    xmax = float(df_x_ref[x_var].max())

    rx_min, rx_max = st.slider(
        f"Rango para {x_var} (cámara {camara_x})",
        xmin, xmax,
        (xmin, xmax)
    )

    # ESTADO DE EJES
    if "axis_frozen_tab2" not in st.session_state:
        st.session_state.axis_frozen_tab2 = False

    if "axis_limits_tab2" not in st.session_state:
        st.session_state.axis_limits_tab2 = {}

    # GRÁFICO
    # APLICAR FILTRO GUARDADO

    fig = go.Figure()
    st.session_state.datos_descarga_tab2 = {}

    for camara, y_vars in y_vars_por_camara.items():

        df_cam = aplicar_exclusion_variables(
            df_camaras_activo[camara],
            camara
        )
    
        df_cam = aplicar_filtros_activos(df_cam, camara)
        st.session_state.datos_descarga_tab2[camara] = []

        # Filtrado por X usando valores de ESA cámara
        if x_var in df_cam.columns:
            df_cam = df_cam[
                (df_cam[x_var] >= rx_min) &
                (df_cam[x_var] <= rx_max)
            ]

        for y in y_vars:
            # CONTROL DE DATAFRAME VACÍO
            if df_cam.empty or df_cam[y].dropna().empty:
                st.info(f"{camara} – {y}: sin datos tras aplicar filtros")
                continue
            
            ymin = float(df_cam[y].min())
            ymax = float(df_cam[y].max())
            
            if np.isnan(ymin) or np.isnan(ymax) or ymin == ymax:
                st.info(f"{camara} – {y}: rango no válido")
                continue
            
            ry_min, ry_max = st.slider(
                f"{camara} – rango {y}",
                ymin,
                ymax,
                (ymin, ymax),
                key=f"slider_{camara}_{y}"
            )

            df_y = df_cam[
                (df_cam[y] >= ry_min) &
                (df_cam[y] <= ry_max)
            ]
            # Guardar datos representados para descarga
            df_export = df_y.copy()
            df_export["Variable_X"] = x_var
            df_export["Variable_Y"] = y
            
            st.session_state.datos_descarga_tab2[camara].append(df_export)

            # SCATTER CON GRADIENTE DE COLOR
            if color_var and color_var in df_y.columns:
            
                fig.add_trace(
                    go.Scatter(
                        x=df_y[x_var] if x_var in df_y.columns else df_y.index,
                        y=df_y[y],
                        mode="markers",
                        name=f"{camara} – {y}",
                        marker=dict(
                            color=df_y[color_var],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=color_var)
                        ),
                        customdata=[(camara, i) for i in df_y.index]
                    )
                )
            
            else:
            
                fig.add_trace(
                    go.Scatter(
                        x=df_y[x_var] if x_var in df_y.columns else df_y.index,
                        y=df_y[y],
                        mode="markers",
                        name=f"{camara} – {y}",
                        customdata=[(camara, i) for i in df_y.index]
                    )
                )

            # Regresión lineal independiente
            if x_var in df_y.columns and len(df_y) >= 2:
                x = df_y[x_var].values
                yy = df_y[y].values

                m, b = np.polyfit(x, yy, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = m * x_line + b

                ss_res = ((yy - (m * x + b)) ** 2).sum()
                ss_tot = ((yy - yy.mean()) ** 2).sum()
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"{camara} – {y} (R²={r2:.3f})"
                    )
                )

    fig.update_layout(
        height=600,
        xaxis_title=f"{x_var} (ref: {camara_x})",
        yaxis_title="Variables",
        legend_title="Cámara / Variable"
    )

    # CONGELAR EJES
    if st.session_state.axis_frozen_tab2:
        fig.update_layout(
            xaxis=dict(range=st.session_state.axis_limits_tab2["x"], autorange=False),
            yaxis=dict(range=st.session_state.axis_limits_tab2["y"], autorange=False)
        )

    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    if not st.session_state.axis_frozen_tab2:
        st.session_state.axis_limits_tab2 = {
            "x": fig.layout.xaxis.range,
            "y": fig.layout.yaxis.range
        }
        st.session_state.axis_frozen_tab2 = True

    # EXCLUSIÓN DESDE GRÁFICO
    if event and event.selection and event.selection.points:
        if st.button("Excluir puntos seleccionados del gráfico"):
            for p in event.selection.points:
                camara, idx = p["customdata"]
                df_cam = df_camaras_activo[camara]

                if idx in df_cam.index:
                    fila = df_cam.loc[[idx]]

                    df_camaras_eliminados[camara] = pd.concat(
                        [df_camaras_eliminados[camara], fila],
                        ignore_index=True
                    )

                    df_camaras_activo[camara] = df_cam.drop(idx)

            st.rerun()

    # EXCLUSIÓN MANUAL POR TABLA
    st.markdown("---")
    st.subheader("Excluir puntos manualmente por cámara")

    for camara in camaras_sel:
        st.markdown(f"**Cámara {camara}**")

        df_tabla = df_camaras_activo[camara].copy()
        df_tabla["Excluir"] = False

        with st.form(f"form_exclusion_{camara}"):
            df_editado = st.data_editor(
                df_tabla,
                num_rows="fixed",
                use_container_width=True
            )
            submit = st.form_submit_button("Excluir filas marcadas")

        if submit:
            filas = df_editado[df_editado["Excluir"]].index.tolist()

            if filas:
                puntos = df_camaras_activo[camara].loc[filas]

                df_camaras_eliminados[camara] = pd.concat(
                    [df_camaras_eliminados[camara], puntos],
                    ignore_index=True
                )

                df_camaras_activo[camara] = df_camaras_activo[camara].drop(filas)
                st.rerun()

    # TABLA DE EXCLUIDOS
    st.markdown("---")
    st.subheader("Puntos excluidos del análisis")

    df_excluidos_total = pd.concat(
        [
            df.assign(Camara=cam)
            for cam, df in df_camaras_eliminados.items()
            if not df.empty
        ],
        ignore_index=True
    ) if any(not df.empty for df in df_camaras_eliminados.values()) else pd.DataFrame()

    if df_excluidos_total.empty:
        st.info("No hay puntos excluidos.")
    else:
        st.dataframe(df_excluidos_total, use_container_width=True)
    from io import BytesIO
    
    st.markdown("---")
    st.subheader("Descargar datos representados")
    
    if any(st.session_state.datos_descarga_tab2.values()):
        output = BytesIO()
    
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for cam, lista in st.session_state.datos_descarga_tab2.items():
                if lista:
                    pd.concat(lista, ignore_index=True).to_excel(
                        writer,
                        sheet_name=cam[:31],
                        index=False
                    )
    
        output.seek(0)
        if st.session_state.filtros_activos:
            nombre = list(st.session_state.filtros_activos)[0]
        else:
            nombre = "sin_filtro"
        
            
        st.download_button(
            "Descargar Excel (una hoja por cámara)",
            data=output,
            file_name=f"datos_grafico_{nombre}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No hay datos para descargar")

# PESTAÑA 3 — VACÍA
with tab3:
    st.subheader("Correlaciones con datos filtrados y ranking configurable")

    # DATOS FILTRADOS
    if not st.session_state.df_camaras_activo:
        st.info("Cargue datos primero")
        st.stop()

    camara = st.selectbox(
        "Cámara",
        list(st.session_state.df_camaras_activo.keys()),
        key="corr_camara"
    )

    df = aplicar_exclusion_variables(
    st.session_state.df_camaras_activo[camara],
    camara
)
    df = aplicar_filtros_activos(df, camara)
    cols_num = df.select_dtypes(include="number").columns.tolist()

    if len(cols_num) < 2:
        st.warning("No hay suficientes variables numéricas tras aplicar filtros")
        st.stop()

    # VARIABLE OBJETIVO
    y_obj = st.selectbox(
        "Variable objetivo",
        cols_num,
        key="corr_y_obj"
    )

    # EXCLUSIÓN DE VARIABLES
    vars_global = st.session_state.variables_excluidas_global.get(camara, [])

    posibles_x = [
        c for c in cols_num
        if c != y_obj and c not in vars_global
    ]


    vars_excluidas = st.multiselect(
        "Variables a excluir del análisis",
        posibles_x,
        default=[],
        key="corr_vars_excluidas"
    )

    X_cols = [c for c in posibles_x if c not in vars_excluidas]

    if len(X_cols) < 1:
        st.warning("No quedan variables para analizar tras la exclusión")
        st.stop()

    # PREPARACIÓN DE DATOS
    df_base = df[X_cols + [y_obj]].dropna()

    if len(df_base) < 20:
        st.warning("Datos insuficientes tras filtros y exclusiones")
        st.stop()

    X = df_base[X_cols]
    y = df_base[y_obj]

    # CÁLCULO DE MÉTRICAS
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LinearRegression
    import numpy as np

    mi = mutual_info_regression(X, y, random_state=42)

    resultados = []

    for i, col in enumerate(X_cols):
        x_col = X[col].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(x_col, y)
        r2 = model.score(x_col, y)

        pearson = df_base[[col, y_obj]].corr().iloc[0, 1]
        spearman = df_base[[col, y_obj]].corr(method="spearman").iloc[0, 1]

        resultados.append({
            "Variable": col,
            "Pearson": pearson,
            "Spearman": spearman,
            "Mutual_Info": mi[i],
            "R2": r2
        })

    df_corr = pd.DataFrame(resultados)

    # NORMALIZACIÓN Y SCORE DE IMPORTANCIA
    def normalizar(s):
        if s.max() == s.min():
            return 0
        return (s - s.min()) / (s.max() - s.min())

    df_corr["Pearson_n"] = normalizar(df_corr["Pearson"].abs())
    df_corr["Spearman_n"] = normalizar(df_corr["Spearman"].abs())
    df_corr["MI_n"] = normalizar(df_corr["Mutual_Info"])
    df_corr["R2_n"] = normalizar(df_corr["R2"])

    df_corr["Score_importancia"] = (
        df_corr["Pearson_n"]
        + df_corr["Spearman_n"]
        + df_corr["MI_n"]
        + df_corr["R2_n"]
    )

    df_corr = (
        df_corr
        .sort_values("Score_importancia", ascending=False)
        .reset_index(drop=True)
    )

    # SELECTOR DE TOP N
    st.markdown("### Configuración del ranking")

    opcion_top = st.selectbox(
        "Mostrar ranking",
        ["Completo", "Top 5", "Top 10"],
        key="corr_top_selector"
    )

    if opcion_top == "Top 5":
        df_rank = df_corr.head(5)
    elif opcion_top == "Top 10":
        df_rank = df_corr.head(10)
    else:
        df_rank = df_corr.copy()

    # TABLA CUANTITATIVA
    st.markdown("### Métricas cuantitativas de correlación")

    st.dataframe(
        df_rank[[
            "Variable",
            "Pearson",
            "Spearman",
            "Mutual_Info",
            "R2",
            "Score_importancia"
        ]].style.background_gradient(
            cmap="RdYlGn",
            subset=["Score_importancia"]
        ),
        use_container_width=True
    )

    # RANKING VISUAL
    st.markdown("### Ranking visual de importancia de correlación")

    fig_rank = px.bar(
        df_rank,
        x="Score_importancia",
        y="Variable",
        orientation="h",
        title="Importancia relativa de las variables respecto a la variable objetivo"
    )

    fig_rank.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Score de importancia (normalizado)",
        yaxis_title=""
    )

    st.plotly_chart(fig_rank, use_container_width=True)

    # VARIABLE MÁS RELEVANTE (TRAS EXCLUSIONES)
    var_top = df_corr.iloc[0]

    st.markdown(
        f"La variable con mayor relación global con **{y_obj}** es "
        f"**{var_top['Variable']}**, considerando las exclusiones aplicadas."
    )

    # MAPA DE CALOR DE CORRELACIONES
    st.markdown("### Mapa de calor de correlaciones (Spearman)")

    corr_matrix = df_base.corr(method="spearman")

    fig_heat = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    # =========================================================
    # EXPORTACIÓN
    # =========================================================
    st.markdown("### Exportación de resultados")

    csv = df_corr.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Descargar ranking completo",
        data=csv,
        file_name=f"ranking_correlaciones_{camara}_{y_obj}.csv",
        mime="text/csv",
        key="corr_download"
    )

# PESTAÑA 4 — VACÍA
with tab4:
    st.subheader("Modelo predictivo: comparación de algoritmos de Machine Learning")

    # DATOS FILTRADOS
    if not st.session_state.df_camaras_activo:
        st.info("Cargue datos primero")
        st.stop()

    camara = st.selectbox(
        "Cámara",
        list(st.session_state.df_camaras_activo.keys()),
        key="model_cmp_camara"
    )

    df = aplicar_exclusion_variables(
    st.session_state.df_camaras_activo[camara],
    camara
)
    df = aplicar_filtros_activos(df, camara)
    cols_num = df.select_dtypes(include="number").columns.tolist()

    if len(cols_num) < 3:
        st.warning("No hay suficientes variables numéricas")
        st.stop()

    # VARIABLE OBJETIVO
    y_obj = st.selectbox(
        "Variable objetivo a predecir",
        cols_num,
        key="model_cmp_y"
    )

    # SELECCIÓN DE NÚMERO DE VARIABLES EXPLICATIVAS
    opcion_vars = st.selectbox(
        "Variables explicativas a utilizar",
        ["Top 5", "Top 10", "Top 15", "Todas"],
        index=1,
        key="model_cmp_topn"
    )


    posibles_x = [c for c in cols_num if c != y_obj]

    # RANKING DE VARIABLES (MISMA LÓGICA QUE CORRELACIONES)
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LinearRegression
    import numpy as np

    df_rank_base = df[posibles_x + [y_obj]].dropna()

    if len(df_rank_base) < 30:
        st.warning("Datos insuficientes para entrenar modelos fiables")
        st.stop()

    X_rank = df_rank_base[posibles_x]
    y_rank = df_rank_base[y_obj]

    mi = mutual_info_regression(X_rank, y_rank, random_state=42)

    ranking = []

    for i, col in enumerate(posibles_x):
        lr = LinearRegression().fit(X_rank[[col]], y_rank)

        ranking.append({
            "Variable": col,
            "Pearson": df_rank_base[[col, y_obj]].corr().iloc[0, 1],
            "Spearman": df_rank_base[[col, y_obj]].corr(method="spearman").iloc[0, 1],
            "Mutual_Info": mi[i],
            "R2": lr.score(X_rank[[col]], y_rank)
        })

    df_rank = pd.DataFrame(ranking)

    def normalizar(s):
        if s.max() == s.min():
            return 0
        return (s - s.min()) / (s.max() - s.min())

    df_rank["Score"] = (
        normalizar(df_rank["Pearson"].abs())
        + normalizar(df_rank["Spearman"].abs())
        + normalizar(df_rank["Mutual_Info"])
        + normalizar(df_rank["R2"])
    )

    df_rank = df_rank.sort_values("Score", ascending=False)

    if opcion_vars == "Top 5":
        X_cols = df_rank.head(5)["Variable"].tolist()
    
    elif opcion_vars == "Top 10":
        X_cols = df_rank.head(10)["Variable"].tolist()
    
    elif opcion_vars == "Top 15":
        X_cols = df_rank.head(15)["Variable"].tolist()
    
    else:
        X_cols = df_rank["Variable"].tolist()
    # EXCLUSIÓN MANUAL DE VARIABLES
    vars_excluir_modelo = st.multiselect(
        "Excluir variables del modelo",
        X_cols,
        default=[],
        key="model_vars_excluir"
    )
    
    # Aplicar exclusión
    X_cols = [v for v in X_cols if v not in vars_excluir_modelo]
    
    if len(X_cols) == 0:
        st.warning("No quedan variables explicativas tras la exclusión")
        st.stop()

    st.write(f"Número de variables utilizadas: {len(X_cols)}")
    st.write("Variables finales del modelo:", X_cols)

    # PREPARACIÓN FINAL DE DATOS
    df_model = df[X_cols + [y_obj]].dropna()

    X = df_model[X_cols]
    y = df_model[y_obj]
    
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

    # DEFINICIÓN DE MODELOS
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

    modelos = {
        "Regresión lineal": LinearRegression(),
    
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
    
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),
    
        "HistGradient Boosting": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        ),
    
        # NUEVOS MODELOS
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
    
        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_state=42
        )
    }


    # ENTRENAMIENTO Y EVALUACIÓN
    resultados = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        resultados[nombre] = {
            "y_pred": y_pred,
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred)
        }

    # MÉTRICAS RESUMEN
    st.markdown("### Métricas comparativas")

    df_metricas = pd.DataFrame({
        nombre: {
            "R2": res["R2"],
            "MAE": res["MAE"]
        }
        for nombre, res in resultados.items()
    }).T

    st.dataframe(df_metricas, use_container_width=True)

    # GRÁFICAS REAL VS PREDICHO (UNA POR MODELO)
    st.markdown("### Comparación visual: valores reales vs predichos")

    cols = st.columns(2)

    for i, (nombre, res) in enumerate(resultados.items()):
        df_plot = pd.DataFrame({
            "Real": y_test,
            "Predicho": res["y_pred"]
        })

        fig = px.scatter(
            df_plot,
            x="Real",
            y="Predicho",
            title=f"{nombre} (R² = {res['R2']:.3f})"
        )

        min_v = min(df_plot.min())
        max_v = max(df_plot.max())

        fig.add_shape(
            type="line",
            x0=min_v, y0=min_v,
            x1=max_v, y1=max_v,
            line=dict(color="black", dash="dash")
        )

        fig.update_layout(
            xaxis_title="Valor real",
            yaxis_title="Valor predicho"
        )

        cols[i % 2].plotly_chart(fig, use_container_width=True)

    # CONCLUSIÓN AUTOMÁTICA
    mejor_modelo = df_metricas["R2"].idxmax()

    st.markdown(
        f"""
        El modelo con mejor rendimiento según R² es **{mejor_modelo}**.
        La comparación se ha realizado usando las mismas variables,
        el mismo conjunto de entrenamiento y el mismo conjunto de validación.
        """
    )

# PESTAÑA 5 
with tab5:
    st.subheader("Simulador de operación basado en Machine Learning")

    # DATOS FILTRADOS
    if not st.session_state.df_camaras_activo:
        st.info("Cargue datos primero")
        st.stop()

    camara = st.selectbox(
        "Cámara",
        list(st.session_state.df_camaras_activo.keys()),
        key="sim_camara"
    )

    df = aplicar_exclusion_variables(
    st.session_state.df_camaras_activo[camara],
    camara
)
    df = aplicar_filtros_activos(df, camara)
    cols_num = df.select_dtypes(include="number").columns.tolist()

    # VARIABLE OBJETIVO
    y_obj = st.selectbox(
        "Variable objetivo a simular",
        cols_num,
        key="sim_y"
    )

    # SELECCIÓN DE VARIABLES TOP N
    opcion_vars = st.selectbox(
        "Variables explicativas",
        ["Top 5", "Top 10", "Top 15", "Todas"],
        index=1,
        key="sim_top"
    )

    posibles_x = [c for c in cols_num if c != y_obj]

    # RANKING DE VARIABLES
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LinearRegression
    import numpy as np

    df_rank_base = df[posibles_x + [y_obj]].dropna()

    X_rank = df_rank_base[posibles_x]
    y_rank = df_rank_base[y_obj]

    mi = mutual_info_regression(X_rank, y_rank, random_state=42)

    ranking = []

    for i, col in enumerate(posibles_x):
        lr = LinearRegression().fit(X_rank[[col]], y_rank)

        ranking.append({
            "Variable": col,
            "Score": (
                abs(df_rank_base[[col, y_obj]].corr().iloc[0, 1])
                + abs(df_rank_base[[col, y_obj]].corr(method="spearman").iloc[0, 1])
                + mi[i]
                + lr.score(X_rank[[col]], y_rank)
            )
        })

    df_rank = pd.DataFrame(ranking).sort_values("Score", ascending=False)

    if opcion_vars == "Top 5":
        X_cols = df_rank.head(5)["Variable"].tolist()
    elif opcion_vars == "Top 10":
        X_cols = df_rank.head(10)["Variable"].tolist()
    elif opcion_vars == "Top 15":
        X_cols = df_rank.head(15)["Variable"].tolist()
    else:
        X_cols = df_rank["Variable"].tolist()

    # EXCLUSIÓN MANUAL
    vars_excluir = st.multiselect(
        "Excluir variables del simulador",
        X_cols,
        default=[],
        key="sim_excluir"
    )

    X_cols = [v for v in X_cols if v not in vars_excluir]

    if len(X_cols) == 0:
        st.warning("No quedan variables para simular")
        st.stop()

    st.write("Variables utilizadas:", X_cols)

    # ENTRENAMIENTO DEL MEJOR MODELO
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor
    )

    df_model = df[X_cols + [y_obj]].dropna()

    X = df_model[X_cols]
    y = df_model[y_obj]
    
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

    modelos = {
        "RF": RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),
    
        "GB": GradientBoostingRegressor(random_state=42),
    
        "HGB": HistGradientBoostingRegressor(random_state=42),
    
        "XGB": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ),
    
        "CAT": CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_state=42
        )
    }


    mejor_modelo = None
    mejor_r2 = -999

    for nombre, m in modelos.items():
        m.fit(X_train, y_train)
        r2 = r2_score(y_test, m.predict(X_test))

        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_modelo = m

    st.success(f"Modelo seleccionado automáticamente (R² = {mejor_r2:.3f})")

    # SLIDERS DE SIMULACIÓN
    st.markdown("### Ajuste de variables operativas")

    entrada_usuario = {}

    for col in X_cols:
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        mean_v = float(df[col].mean())

        entrada_usuario[col] = st.slider(
            col,
            min_v,
            max_v,
            mean_v,
            key=f"sim_slider_{col}"
        )

    entrada_df = pd.DataFrame([entrada_usuario])

    # PREDICCIÓN
    pred = mejor_modelo.predict(entrada_df)[0]

    st.metric("Predicción de la variable objetivo", f"{pred:.3f}")

    # COMPARACIÓN CON MEDIDA REAL
    valor_real = st.number_input(
        "Introducir valor real medido (opcional)",
        value=0.0,
        key="sim_valor_real"
    )

    if valor_real != 0:
        error = pred - valor_real
        st.metric("Error de predicción", f"{error:.3f}")

    # POSICIÓN DENTRO DEL HISTÓRICO
    hist_min = df[y_obj].min()
    hist_max = df[y_obj].max()

    st.markdown("### Posición dentro del histórico")

    st.progress(
        float((pred - hist_min) / (hist_max - hist_min))
    )

    # GRÁFICA COMPARATIVA
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Box(
        y=df[y_obj],
        name="Histórico"
    ))

    fig.add_trace(go.Scatter(
        y=[pred],
        x=["Predicción"],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="Predicción"
    ))

    st.plotly_chart(fig, use_container_width=True)
