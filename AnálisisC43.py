import streamlit as st
import pandas as pd
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
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

# 🔥 APLASTAR COLUMNAS (MUY IMPORTANTE)
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

st.success(f"Datos cargados: {len(df)} filas | {len(variables)} variables")

# ============================================
# TABS
# ============================================

tab1, tab2, tab3 = st.tabs([
    "Graficado",
    "Ranking de correlaciones",
    "Mapa de correlaciones"
])

# ============================================
# TAB 1 — GRAFICADO
# ============================================

with tab1:

    st.subheader("Graficado interactivo")

    x_var = st.selectbox(
        "Variable eje X",
        variables
    )

    y_vars = st.multiselect(
        "Variables Y",
        [v for v in variables if v != x_var],
        default=[v for v in variables if v != x_var][:2]
    )

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
            (ymin, ymax)
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

        df_plot = df[[x_var, y]].dropna()
        for y in y_vars:
        
            df_plot = df[[x_var, y]].dropna()
        
            # ✅ DEBUG AQUÍ
            st.write(f"{y} → puntos:", len(df_plot))
            if len(df_plot) < 3:
                continue
        if df_plot.empty:
            continue

        if color_var and color_var in df_filt.columns:

            fig.add_trace(
                go.Scatter(
                    x=df_filt[x_var],
                    y=df_filt[y],
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
                    x=df_filt[x_var],
                    y=df_filt[y],
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

    y_obj = st.selectbox(
        "Variable objetivo",
        variables
    )

    x_rank = [v for v in variables if v != y_obj]

    resultados = []

    y_series = df[y_obj]

    if not isinstance(y_series, pd.Series):
        st.warning("Variable objetivo no válida")
        st.stop()
    
    y_series = pd.to_numeric(y_series, errors="coerce")
    
    for col in x_rank:
    
        try:
            x_series = df[col]
            
            # si no es serie, saltar
            if not isinstance(x_series, pd.Series):
                continue
        
            x_series = pd.to_numeric(x_series, errors="coerce")
        
        except:
            continue
            
        df_temp = pd.DataFrame({
            "x": x_series,
            "y": y_series
        }).dropna()
    
        # basta con pocos puntos
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
            "Pearson": pearson,
            "Spearman": spearman,
            "Mutual_Info": mi,
            "R2": r2,
            "Score": score
        })
    
    if len(resultados) == 0:
        st.warning("No hay suficientes datos para calcular correlaciones")
    else:
        df_rank = pd.DataFrame(resultados).sort_values("Score", ascending=False)
        
        st.dataframe(df_rank)
    
        fig_rank = px.bar(
            df_rank,
            x="Score",
            y="Variable",
            orientation="h",
            color="Score",
            color_continuous_scale="YlOrBr"
        )
    
        fig_rank.update_layout(
            yaxis=dict(autorange="reversed")
        )
    
        st.plotly_chart(fig_rank, use_container_width=True)

# ============================================
# TAB 3 — HEATMAP
# ============================================

with tab3:

    st.subheader("Mapa de correlaciones")

    corr = df[variables].corr(method="spearman")

    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    st.plotly_chart(fig_heat, use_container_width=True)