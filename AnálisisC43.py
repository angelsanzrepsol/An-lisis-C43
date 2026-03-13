import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

st.set_page_config(
    page_title="Análisis C43",
    layout="wide"
)

# ============================================
# ESTILO INDUSTRIAL (MISMO QUE TU APP)
# ============================================

st.markdown("""
<style>

/* Fondo general */
html, body, .block-container, [class*="stApp"] {
    background-color: #FFFFFF !important;
    color: #333333 !important;
}

/* Títulos */
h1, h2, h3, h4 {
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
    color: #D98B3B !important;
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

# ============================================
# TÍTULO
# ============================================

st.markdown(
    "<h1 class='darkblue-title'>Analizador interactivo de variables de proceso</h1>",
    unsafe_allow_html=True
)

# ============================================
# CARGA DE DATOS
# ============================================

st.sidebar.header("Carga de datos")

file = st.sidebar.file_uploader(
    "Subir Excel o CSV",
    type=["xlsx","xls","csv"]
)

if file is None:
    st.info("Suba un archivo para comenzar")
    st.stop()

if file.name.endswith(".csv"):
    df = pd.read_csv(file)
else:
    df = pd.read_excel(file)

df = df.select_dtypes(include="number")

st.success(f"Datos cargados: {df.shape[0]} filas | {df.shape[1]} variables")

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

    st.subheader("Graficado interactivo de variables")

    cols = df.columns.tolist()

    x_var = st.selectbox(
        "Variable eje X",
        cols
    )

    y_vars = st.multiselect(
        "Variables Y a graficar",
        [c for c in cols if c != x_var],
        default=[c for c in cols if c != x_var][:2]
    )

    color_var = st.selectbox(
        "Variable para gradiente de color",
        ["(ninguna)"] + cols
    )

    if color_var == "(ninguna)":
        color_var = None

    if len(y_vars) == 0:
        st.warning("Seleccione al menos una variable Y")
        st.stop()

    # =====================================
    # FILTROS
    # =====================================

    st.markdown("### Rangos de variables")

    xmin = float(df[x_var].min())
    xmax = float(df[x_var].max())

    rx = st.slider(
        f"Rango {x_var}",
        xmin,
        xmax,
        (xmin,xmax)
    )

    df_filt = df[(df[x_var]>=rx[0]) & (df[x_var]<=rx[1])]

    rangos_y = {}

    for y in y_vars:

        ymin = float(df_filt[y].min())
        ymax = float(df_filt[y].max())

        r = st.slider(
            f"Rango {y}",
            ymin,
            ymax,
            (ymin,ymax)
        )

        rangos_y[y] = r

    for y,r in rangos_y.items():

        df_filt = df_filt[
            (df_filt[y]>=r[0]) &
            (df_filt[y]<=r[1])
        ]

    st.write("Datos tras filtros:",df_filt.shape[0])

    # =====================================
    # GRÁFICO
    # =====================================

    fig = go.Figure()

    for y in y_vars:

        df_plot = df_filt[[x_var,y]].dropna()

        if df_plot.empty:
            continue

        # scatter

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

        # regresión

        if len(df_plot)>2:

            x = df_plot[x_var].values
            yy = df_plot[y].values

            model = LinearRegression()
            model.fit(x.reshape(-1,1),yy)

            r2 = model.score(x.reshape(-1,1),yy)

            x_line = np.linspace(x.min(),x.max(),100)
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

    st.plotly_chart(fig,use_container_width=True)

# ============================================
# TAB 2 — RANKING
# ============================================

with tab2:

    st.subheader("Ranking de correlaciones")

    cols = df.columns.tolist()

    y_obj = st.selectbox(
        "Variable objetivo",
        cols
    )

    x_rank = [c for c in cols if c != y_obj]

    df_clean = df[x_rank+[y_obj]].dropna()

    X = df_clean[x_rank]
    y = df_clean[y_obj]

    mi = mutual_info_regression(X,y)

    resultados=[]

    for i,col in enumerate(x_rank):

        x = df_clean[col].values.reshape(-1,1)

        model = LinearRegression()
        model.fit(x,y)

        r2 = model.score(x,y)

        pearson = df_clean[[col,y_obj]].corr().iloc[0,1]
        spearman = df_clean[[col,y_obj]].corr(method="spearman").iloc[0,1]

        score = abs(pearson)+abs(spearman)+mi[i]+r2

        resultados.append({
            "Variable":col,
            "Pearson":pearson,
            "Spearman":spearman,
            "Mutual_Info":mi[i],
            "R2":r2,
            "Score":score
        })

    df_rank = pd.DataFrame(resultados).sort_values("Score",ascending=False)

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

    st.plotly_chart(fig_rank,use_container_width=True)

# ============================================
# TAB 3 — HEATMAP
# ============================================

with tab3:

    st.subheader("Mapa de correlaciones")

    corr = df.corr(method="spearman")

    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    st.plotly_chart(fig_heat,use_container_width=True)