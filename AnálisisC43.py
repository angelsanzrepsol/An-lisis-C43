import streamlit as st
import pandas as pd
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from streamlit_plotly_events import plotly_events
import json

# ============================================
# FUNCIONES AUXILIARES (NUEVO)
# ============================================

def aplicar_filtro(df_base, filtro):
    df_f = df_base.copy()

    if "fecha" in filtro:
        f_ini = pd.to_datetime(filtro["fecha"][0])
        f_fin = pd.to_datetime(filtro["fecha"][1])

        df_f = df_f[
            (df_f["Fecha"] >= f_ini) &
            (df_f["Fecha"] <= f_fin)
        ]

    for var, (vmin, vmax) in filtro["rangos"].items():
        if var in df_f.columns:
            df_f = df_f[
                (df_f[var] >= vmin) &
                (df_f[var] <= vmax)
            ]

    df_f = df_f.drop(
        index=filtro.get("excluidos", []),
        errors="ignore"
    )

    return df_f


def calcular_ranking(df_rank_base, y_obj, x_rank):

    resultados = []

    y_series = pd.to_numeric(df_rank_base[y_obj], errors="coerce")

    for col in x_rank:

        x_series = pd.to_numeric(df_rank_base[col], errors="coerce")

        df_temp = pd.DataFrame({
            "x": x_series,
            "y": y_series
        }).dropna()

        if len(df_temp) < 5:
            continue

        X = df_temp["x"].values.reshape(-1,1)
        Y = df_temp["y"].values

        try:
            model = LinearRegression()
            model.fit(X, Y)
            r2 = model.score(X, Y)
        except:
            r2 = 0

        pearson = df_temp["x"].corr(df_temp["y"])
        spearman = df_temp["x"].corr(df_temp["y"], method="spearman")

        try:
            mi = mutual_info_regression(X, Y)[0]
        except:
            mi = 0

        score = abs(pearson) + abs(spearman) + r2 + mi

        resultados.append({
            "Variable": col,
            "Score": score
        })
    
    df_res = pd.DataFrame(resultados)
    
    if df_res.empty or "Score" not in df_res.columns:
        return pd.DataFrame(columns=["Variable", "Score"])
    
    return df_res.sort_values("Score", ascending=False)
def construir_columnas(df_raw, n_header_rows):
    headers = [df_raw.iloc[i] for i in range(n_header_rows)]

    cols = []

    for col_idx in range(len(df_raw.columns)):

        partes = []

        for h in headers:
            val = h[col_idx]
            if pd.notna(val):
                partes.append(str(val))

        if col_idx == 0:
            cols.append("Fecha")
        elif col_idx == 1:
            cols.append("Estado")
        else:
            if len(partes) == 0:
                cols.append(f"Var_{col_idx}")
            else:
                cols.append(" | ".join(partes))

    return cols
    # convertir todo a numérico (excepto Fecha)
    for c in df.columns:
        if c != "Fecha":
            try:
                df[c] = pd.to_numeric(df[c].squeeze(), errors="coerce")
            except:
                pass
# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

st.set_page_config(
    page_title="Analizador de proceso",
    layout="wide"
)

# ============================================
# ESTILO INDUSTRIAL
# ============================================

st.markdown("""
<style>

html, body, .block-container, [class*="stApp"] {
    background-color: #FFFFFF !important;
    color: #333333 !important;
}

h1, h2, h3, h4 {
    color: #D98B3B !important;
    font-weight: 800 !important;
}

.darkblue-title {
    color: #0B1A33 !important;
    font-weight: 800 !important;
}

.stTabs [data-baseweb="tab"] p {
    color: #666666 !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] p {
    color: #D98B3B !important;
    font-weight: 700 !important;
}

.stButton>button {
    background-color: #D98B3B !important;
    color: white !important;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# TÍTULO
# ============================================

st.markdown(
    "<h1 class='darkblue-title'>Análisis C43</h1>",
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
# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Carga de datos")

file = st.sidebar.file_uploader(
    "Subir archivo",
    type=["xlsx","xls","csv"]
)
# ============================================
# ESTADOS DE FILTROS
# ============================================

if "filtros_guardados" not in st.session_state:
    st.session_state.filtros_guardados = {}
    
if file is None:
    st.info("Suba un archivo para comenzar")
    st.stop()

# ============================================
# LECTURA DE EXCEL
# ============================================

xls = pd.ExcelFile(file)
# ============================================
# FILTROS GLOBALES DESDE GENERAL
# ============================================

df_general_raw = pd.read_excel(xls, sheet_name="General", header=None)

cols_general = construir_columnas(df_general_raw, 5)

df_general = df_general_raw.iloc[5:].copy()
df_general.columns = cols_general

df_general["Fecha"] = pd.to_datetime(df_general["Fecha"], errors="coerce").dt.normalize()
df_general["Estado"] = df_general["Estado"].astype(str).str.upper()

# -------- SELECTORES --------

estado_sel = st.sidebar.selectbox(
    "Estado planta",
    ["TODOS", "MARCHA", "PARO"]
)

modo_prod = st.sidebar.selectbox(
    "Modo producción",
    ["TOTAL", "MONOPRODUCCIÓN", "COPRODUCCIÓN"]
)

sheets_sel = st.sidebar.multiselect(
    "Seleccionar pestañas",
    xls.sheet_names,
    default=xls.sheet_names[:2]
)

# -------- FILTRO ESTADO --------

if estado_sel != "TODOS":
    fechas_estado = df_general[
        df_general["Estado"].str.contains(estado_sel)
    ]["Fecha"]
else:
    fechas_estado = df_general["Fecha"]

# -------- FILTRO PRODUCCIÓN --------

col3 = df_general.columns[2]

if modo_prod == "MONOPRODUCCIÓN":
    fechas_prod = df_general[
        df_general[col3].astype(str).str.contains("ACV-DIESEL", na=False)
    ]["Fecha"]

elif modo_prod == "COPRODUCCIÓN":
    fechas_prod = df_general[
        df_general[col3].astype(str).str.contains("ACV-COPROD|PFAD-COPROD", na=False)
    ]["Fecha"]

else:
    fechas_prod = df_general["Fecha"]

# -------- COMBINAR --------

fechas_validas = set(fechas_estado) & set(fechas_prod)

# ============================================
# LEER TODAS LAS PESTAÑAS SELECCIONADAS
# ============================================

dfs = []

for sh in sheets_sel:

    df_raw = pd.read_excel(xls, sheet_name=sh, header=None)

    if sh == "General":
        n_header = 5
    elif "SGL" in sh:
        n_header = 3
    else:
        n_header = 3

    cols = construir_columnas(df_raw, n_header)

    df_tmp = df_raw.iloc[n_header:].copy()
    df_tmp.columns = cols

    df_tmp["Fecha"] = pd.to_datetime(df_tmp["Fecha"], errors="coerce").dt.normalize()

    df_tmp = df_tmp[df_tmp["Fecha"].isin(fechas_validas)]

    df_tmp = df_tmp.set_index("Fecha")
    df_tmp = df_tmp.add_prefix(f"{sh} | ")

    dfs.append(df_tmp)

df = pd.concat(dfs, axis=1)

# APLASTAR COLUMNAS 
df.columns = [str(c) for c in df.columns]

# eliminar duplicadas
df = df.loc[:, ~df.columns.duplicated()]

df = df.reset_index()
# ============================================
# LIMPIEZA FINAL (MUY IMPORTANTE)
# ============================================

for c in df.columns:
    if c != "Fecha":
        try:
            df[c] = pd.to_numeric(df[c].squeeze(), errors="coerce")
        except:
            pass

variables = [c for c in df.columns if c != "Fecha"]
# ============================================
# GRUPOS DE VARIABLES
# ============================================

GRUPOS = {

    # ---------------- GENERAL ----------------
    "ALIMENTACIÓN": [
        "607FI0007","607FI0041","607FC0001","607PDI002","607TI0004",
        "607TC0007","607FC0010","607FI0033","607FI0090","607FI0032",
        "607FC0091","607TI0090","607AI001A","607AI001B","607AI001C"
    ],

    "REACTOR C-02": [
        "607FC0011","607FFC014","607FC0014","607FC0020","607FFC015",
        "607FC0015","607FFI151","607FFI156","607PI0031","607PI0035",
        "607PI0036","607PI0032","607PDI034","607PDI033","607PDI037",
        "607TI0048","607TC0052","607TI0012","607TDI031","607TI0014",
        "607TDI032","607TI0041"
    ],

    "SEPARACIÓN HDT": [
        "607TI0015","607TC0061","607TI0059","607PI0042","607FC0016",
        "607DI0001","607TI0065","607PC055B","607FC0023","607FI0024"
    ],

    "ABSORBER AMINAR": [
        "607FC0027","607FI0115","607TDI071","607PDI061","607PI0060","607FC0030"
    ],

    "STRIPPER C-008 / C-009": [
        "607TI0102","607FC0034","607PC0087","607TI0100","607FC0038",
        "607PI0096","607TI0103","607FI0037","607TI0099","607FC0043"
    ],

    "REACTOR C-12": [
        "607TI0119","607TI0121","607TI0420","607TI0421","607PI0118",
        "607TC0147","607TI0145","607TDI131","607TI0148","607PI0120",
        "607PI0119","607PDI121","607FFI157"
    ],

    "SEPARACIÓN HDI": [
        "607TC0111","607TI0161","607PI0124","607FC0047","607TI0163",
        "607PC129A","607FC0048","607FC0049"
    ],

    "STRIPPER C-016/C-017": [
        "607TI0171","607FC0051","607TI0169","607PC0131","607PC0131.OP",
        "607TI0170","607FC0055","607PI0138","607TI0177","607FI0053",
        "607TI0172","607FC0052"
    ],

    "STRIPPER C-18/C-19": [
        "607TI0187","607TI0186","607TI0188","607TI0183","607TI0185",
        "607TC0189","607TI0184","607TI0180","607TI0460","607TI0461",
        "607FC0056","607FC0101","607PC0149","607PI0147","607PDI151",
        "607PI0154","607FC0059","607DUTYF2","607PV149B"
    ],

    "EXTRACCIONES": [
        "607FC0061","607FC0061.OP","607FC0062","607FC0062.OP",
        "607FC0067","607FC0067.OP"
    ]
}
# ============================================
# EXPANDIR GRUPOS → VARIABLES REALES
# ============================================

def expandir_grupos(seleccion, variables_df):
    resultado = []

    for item in seleccion:

        # si es grupo
        if item in GRUPOS:
            tags = GRUPOS[item]

            for col in variables_df:
                for tag in tags:
                    if tag in col.upper():
                        resultado.append(col)

        # si es variable normal
        else:
            resultado.append(item)

    return list(set(resultado))
# ============================================
# MAPA JERÁRQUICO DE VARIABLES
# ============================================

mapa_variables = {}

for col in variables:
    
    partes = col.split(" | ")
    
    if len(partes) < 2:
        continue
    
    nivel1 = partes[1] if len(partes) > 1 else "Otros"
    nivel2 = partes[2] if len(partes) > 2 else "General"

    if nivel1 not in mapa_variables:
        mapa_variables[nivel1] = {}

    if nivel2 not in mapa_variables[nivel1]:
        mapa_variables[nivel1][nivel2] = []

    mapa_variables[nivel1][nivel2].append(col)
st.success(f"Datos cargados: {len(df)} filas | {len(variables)} variables")

# ============================================
# TABS
# ============================================

tab_filtros, tab1, tab2, tab3 = st.tabs([
    "Creador de filtros",
    "Graficado",
    "Ranking de correlaciones",
    "Mapa de correlaciones"
])


with tab_filtros:

    st.subheader("Creador de filtros avanzado")
    
    if df.empty:
        st.warning("No hay datos")
        st.stop()

    # ===============================
    # VARIABLES
    # ===============================
    variables_all = df.columns.tolist()

    # ============================================
    # SELECCIÓN MULTIVARIABLE
    # ============================================
    
    variables_all = [c for c in df.columns if c != "Fecha"]
    
    opciones = list(GRUPOS.keys()) + variables_all
    
    seleccion = st.multiselect(
        "Variables o grupos",
        opciones,
        key="filtros_vars"
    )
    
    vars_sel = expandir_grupos(seleccion, variables_all)
    
    if len(vars_sel) < 1:
        st.warning("Selecciona al menos una variable")
        st.stop()
    st.markdown("### Filtro por fecha")
    # asegurar tipo datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    
    fecha_min = df["Fecha"].min().to_pydatetime()
    fecha_max = df["Fecha"].max().to_pydatetime()
        
    rango_fecha = st.slider(
            "Rango de fechas",
            min_value=fecha_min,
            max_value=fecha_max,
            value=(fecha_min, fecha_max),
            format="YYYY-MM-DD"
        )
    
    df_work = df[
        (df["Fecha"] >= rango_fecha[0]) &
        (df["Fecha"] <= rango_fecha[1])
    ].copy()
    df = df.copy() 
    df["Fecha_num"] = df["Fecha"].astype("int64") / 1e9
    # ===============================
    # SLIDERS
    # ===============================
    filtro_temp = {}

    for var in vars_sel:
    
        serie = df[var].dropna()
    
        if serie.empty:
            continue
    
        vmin = float(serie.min())
        vmax = float(serie.max())
    
        if vmin == vmax:
            continue
    
        r = st.slider(
            f"Rango {var}",
            vmin,
            vmax,
            (vmin, vmax),
            key=f"multi_{var}"
        )
    
        filtro_temp[var] = r
    df_work = df[
        (df["Fecha"] >= rango_fecha[0]) &
        (df["Fecha"] <= rango_fecha[1])
    ].copy()
    
    for var, (vmin, vmax) in filtro_temp.items():
        df_work = df_work[
            (df_work[var] >= vmin) &
            (df_work[var] <= vmax)
        ]
    
    # ===============================
    # ESTADO DE EXCLUSIÓN
    # ===============================
    if "puntos_excluidos" not in st.session_state:
        st.session_state.puntos_excluidos = set()
        
    st.markdown("### Visualización")
    # ===============================
    # ===============================
    # SELECCIÓN PARA GRAFICAR
    # ===============================
    
    if len(vars_sel) < 2:
        st.warning("Selecciona al menos 2 variables")
        st.stop()
    
    x_plot = st.selectbox(
        "Variable eje X",
        ["Fecha"] + vars_sel
    )

    y_vars_plot = st.multiselect(
        "Variables eje Y",
        [v for v in vars_sel if v != x_plot],
        default=[v for v in vars_sel if v != x_plot][:2]
    )
    fig = go.Figure()
    
    df_plot = df_work.copy()
    
    # aplicar exclusión previa
    if len(st.session_state.puntos_excluidos) > 0:
        df_plot = df_plot[~df_plot.index.isin(st.session_state.puntos_excluidos)]
    
    # crear trazas
    for var in y_vars_plot:
    
        df_temp = df_plot[[x_plot, var]].dropna()
    
        if df_temp.empty:
            continue
    
        fig.add_trace(
            go.Scatter(
                x=df_temp[x_plot],
                y=df_temp[var],
                mode="markers",
                name=var,
                customdata=df_temp.index  
            )
        )
    
    # MOSTRAR GRÁFICA + CAPTURAR SELECCIÓN
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun"
    )
    
    # BOTÓN EXCLUIR
    if event and event.selection and event.selection.points:
    
        if st.button("Excluir puntos seleccionados"):
    
            for p in event.selection.points:
                idx = p["customdata"]
    
                st.session_state.puntos_excluidos.add(idx)
    
            st.rerun()
  
    # aplicar exclusión
    df_filtrado = df_work.drop(
        index=st.session_state.puntos_excluidos,
        errors="ignore"
    )
    st.write("Puntos tras exclusión:", len(df_filtrado))

    # ===============================
    # GUARDAR FILTRO
    # ===============================
    nombre = st.text_input("Nombre del filtro")

    if st.button("Guardar filtro"):

        st.session_state.filtros_guardados[nombre] = {
            "rangos": filtro_temp,
            "fecha": [str(rango_fecha[0]), str(rango_fecha[1])],
            "excluidos": list(st.session_state.puntos_excluidos)
        }

        st.success(f"Filtro '{nombre}' guardado")
        st.markdown("### Filtros disponibles")
        
        if not st.session_state.filtros_guardados:
            st.info("No hay filtros guardados")
        
        else:
            for nombre, f in st.session_state.filtros_guardados.items():
        
                with st.expander(nombre):
        
                    st.write("Rangos:")
                    st.json(f["rangos"])
        
                    if st.button("Eliminar", key=f"del_{nombre}"):
                        del st.session_state.filtros_guardados[nombre]
                        st.rerun()
    st.markdown("### Exportar filtros")

    if st.session_state.filtros_guardados:
    
        filtros_json = json.dumps(
            st.session_state.filtros_guardados,
            indent=4
        )
    
        st.download_button(
            " Descargar filtros",
            data=filtros_json,
            file_name="filtros_c43.json",
            mime="application/json"
        )
        
    st.markdown("### Importar filtros")

    filtro_file = st.file_uploader(
        "Subir archivo JSON",
        type=["json"]
    )
    if filtro_file is None:
        st.session_state.filtro_importado = False
    if "filtro_importado" not in st.session_state:
        st.session_state.filtro_importado = False
    
    if filtro_file is not None and not st.session_state.filtro_importado:
        try:
            filtros_importados = json.load(filtro_file)
    
            if isinstance(filtros_importados, dict):
    
                for nombre, filtro in filtros_importados.items():
                    st.session_state.filtros_guardados[nombre] = filtro
    
                st.session_state.filtro_importado = True
                st.success("Filtros importados correctamente")
    
            else:
                st.error("Formato incorrecto")
                
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
# ============================================
# TAB 1 — GRAFICADO
# ============================================

with tab1:

    st.subheader("Graficado interactivo")

    x_var = st.selectbox(
        "Variable eje X",
        variables
    )

    opciones = list(GRUPOS.keys()) + variables

    seleccion_y = st.multiselect(
        "Variables Y o grupos",
        opciones,
        key="grafico_vars"
    )
    
    y_vars = expandir_grupos(seleccion_y, variables)
    color_var = st.selectbox(
        "Gradiente de color",
        ["(ninguna)"] + variables
    )

    if color_var == "(ninguna)":
        color_var = None

    if len(y_vars) == 0:
        st.warning("Seleccione al menos una variable")
        st.stop()

    # ============================================
    # FILTROS
    # ============================================

    st.markdown("### Filtros")
    
    if df[x_var].dropna().empty:
        st.warning("Variable sin datos válidos")
        st.stop()
    serie = df[x_var].dropna()

    if serie.empty:
        st.warning(f"{x_var} no tiene datos válidos")
        st.stop()
    
    xmin = float(serie.min())
    xmax = float(serie.max())
    
    # evitar caso valores iguales
    if xmin == xmax:
        st.warning(f"{x_var} tiene valor constante")
        st.stop()

    rx = st.slider(
        f"Rango {x_var}",
        xmin,
        xmax,
        (xmin, xmax)
    )
    # ============================================
    # APLICAR FILTRO GUARDADO
    # ============================================
    filtro_sel = st.selectbox(
        "Filtro guardado",
        ["(ninguno)"] + list(st.session_state.filtros_guardados.keys())
    )
    
    df_filtrado = df.copy()

    if filtro_sel != "(ninguno)":
        # aplicar fecha
        if "fecha" in f:
            f_ini = pd.to_datetime(f["fecha"][0])
            f_fin = pd.to_datetime(f["fecha"][1])
        
            df_filtrado = df_filtrado[
                (df_filtrado["Fecha"] >= f_ini) &
                (df_filtrado["Fecha"] <= f_fin)
            ]
        
        f = st.session_state.filtros_guardados[filtro_sel]
    
        for var, (vmin, vmax) in f["rangos"].items():
    
            if var in df_filtrado.columns:
    
                df_filtrado = df_filtrado[
                    (df_filtrado[var] >= vmin) &
                    (df_filtrado[var] <= vmax)
                ]
    
        df_filtrado = df_filtrado.drop(
            index=f.get("excluidos", []),
            errors="ignore"
        )
    df_filt = df[(df[x_var] >= rx[0]) & (df[x_var] <= rx[1])]

    rangos_y = {}

    for y in y_vars:

        serie_y = df_filt[y].dropna()

        if serie_y.empty:
            continue
        
        ymin = float(serie_y.min())
        ymax = float(serie_y.max())
        
        if ymin == ymax:
            continue

        r = st.slider(
            f"Rango {y}",
            ymin,
            ymax,
            (ymin, ymax),
            key=f"slider_grafico_{y}"
        )

        rangos_y[y] = r

    for y, r in rangos_y.items():

        df_filt = df_filt[
            (df_filt[y] >= r[0]) &
            (df_filt[y] <= r[1])
        ]

    st.write("Filas tras filtros:", df_filt.shape[0])

    # ============================================
    # GRÁFICO
    # ============================================

    fig = go.Figure()

    for y in y_vars:

        df_plot = df_filtrado[[x_var, y]].dropna()
    
        # DEBUG
        st.write(f"{y} → puntos:", len(df_plot))
    
        if len(df_plot) < 3:
            continue
        if df_plot.empty:
            continue

        if color_var and color_var in df_filt.columns:

            fig.add_trace(
                go.Scatter(
                    x=df_plot[x_var],
                    y=df_plot[y],
                    mode="markers",
                    name=y,
                    marker=dict(
                        color=df_filt[color_var],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=color_var)
                    )
                )
            )

        else:

            fig.add_trace(
                go.Scatter(
                    x=df_plot[x_var],
                    y=df_plot[y],
                    mode="markers",
                    name=y
                )
            )

        if len(df_plot) > 2:

            x = df_plot[x_var].values
            yy = df_plot[y].values

            model = LinearRegression()
            model.fit(x.reshape(-1,1), yy)

            r2 = model.score(x.reshape(-1,1), yy)

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = model.predict(x_line.reshape(-1,1))

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"{y} (R²={r2:.3f})"
                )
            )

    fig.update_layout(
        height=700,
        xaxis_title=x_var,
        yaxis_title="Variables",
        legend_title="Variables"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2 — RANKING
# ============================================

with tab2:

    st.subheader("Ranking de correlaciones")
    filtros_sel = st.multiselect(
        "Filtros a comparar",
        list(st.session_state.filtros_guardados.keys())
    )
  
    y_obj = st.selectbox(
        "Variable objetivo",
        variables
    )
    opciones = list(GRUPOS.keys()) + variables

    seleccion_rank = st.multiselect(
        "Variables o grupos",
        opciones,
        key="ranking_vars"
    )
    
    variables_rank = expandir_grupos(seleccion_rank, variables)
    x_rank = [v for v in variables_rank if v != y_obj]

    # ============================================
    # ESCENARIOS (GLOBAL + FILTROS)
    # ============================================
    
    escenarios = {}
    
    # GLOBAL
    escenarios["GLOBAL"] = df.copy()
    
    # FILTROS
    for f_name in filtros_sel:
        f = st.session_state.filtros_guardados[f_name]
        escenarios[f_name] = aplicar_filtro(df, f)
    
    # ============================================
    # CALCULAR RANKINGS
    # ============================================
    
    df_compare = None
    
    for nombre, df_esc in escenarios.items():
    
        if len(df_esc) < 5:
            continue
    
        df_rank = calcular_ranking(df_esc, y_obj, x_rank)
    
        df_rank = df_rank.rename(columns={"Score": nombre})
    
        if df_compare is None:
            df_compare = df_rank
        else:
            df_compare = df_compare.merge(df_rank, on="Variable", how="outer")
    
    # ============================================
    # DESVIACIÓN VS GLOBAL
    # ============================================
    
    if df_compare is not None:
        for f_name in filtros_sel:
            if f_name in df_compare.columns:
                df_compare[f"Δ {f_name}"] = df_compare[f_name] - df_compare["GLOBAL"]
    
    # ============================================
    # MOSTRAR RESULTADOS
    # ============================================
    
    if df_compare is None:
        st.warning("No hay datos suficientes")
    else:
    
        st.dataframe(df_compare)
        
        # ============================================
        # FORMATO LARGO PARA PLOTLY (SOLUCIÓN)
        # ============================================
        
        cols_plot = ["GLOBAL"] + filtros_sel
        cols_plot = [c for c in cols_plot if c in df_compare.columns]
        
        df_melt = df_compare.melt(
            id_vars="Variable",
            value_vars=cols_plot,
            var_name="Escenario",
            value_name="Score"
        )
        
        df_melt["Score"] = pd.to_numeric(df_melt["Score"], errors="coerce")
        
        fig = px.bar(
            df_melt,
            y="Variable",
            x="Score",
            color="Escenario",
            barmode="group"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        # ============================================
        # IMPACTO DE FILTROS (Δ vs GLOBAL)
        # ============================================
        
        cols_delta = [f"Δ {f}" for f in filtros_sel if f"Δ {f}" in df_compare.columns]
        
        if cols_delta:
        
            df_delta = df_compare.melt(
                id_vars="Variable",
                value_vars=cols_delta,
                var_name="Filtro",
                value_name="Delta"
            )
        
            df_delta["Delta"] = pd.to_numeric(df_delta["Delta"], errors="coerce")
        
            st.markdown("### Impacto de filtros (Δ vs GLOBAL)")
        
            fig_delta = px.bar(
                df_delta,
                y="Variable",
                x="Delta",
                color="Filtro",
                barmode="group"
            )
        
            st.plotly_chart(fig_delta, use_container_width=True)

# ============================================
# TAB 3 — HEATMAP
# ============================================

with tab3:

    st.subheader("Mapa de correlaciones")
    opciones = list(GRUPOS.keys()) + variables

    seleccion_heat = st.multiselect(
        "Variables o grupos",
        opciones,
        key="heatmap_vars"
    )
    
    vars_heatmap = expandir_grupos(seleccion_heat, variables)
    corr = df[vars_heatmap].corr(method="spearman")

    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    st.plotly_chart(fig_heat, use_container_width=True)