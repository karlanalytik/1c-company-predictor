import json

import altair as alt
import awswrangler as wr
import boto3
import pandas as pd
import psycopg2
import streamlit as st


BUCKET_NAME = "itam-analytics-paulo"
BASE_PATH = f"s3://{BUCKET_NAME}/sales_predict"
GOLD_PATH = f"{BASE_PATH}/gold/modeling_sales/"
FORECAST_PATH = f"{BASE_PATH}/predictions/xgboost_forecast/"
METRICS_PATH = f"{BASE_PATH}/artifacts/xgboost_simple_global.csv"
SECRET_NAME = "proyecto_aprofundo"
AWS_REGION = "us-east-1"
RDS_HOST = "streamlit-db.csfg0ogwqf7m.us-east-1.rds.amazonaws.com"
RDS_DATABASE = "postgres"


st.set_page_config(
    page_title="Seguimiento y predicción de ventas",
    page_icon="📈",
    layout="wide"
)


st.title("Seguimiento y predicción de ventas")
st.write(
    "Aplicación para explorar ventas históricas, consultar proyecciones "
    "y registrar retroalimentación de negocio."
)


GOLD_COLUMNS = [
    "shop_id",
    "item_id",
    "item_name",
    "shop_name",
    "item_category_name",
    "date_block_num",
    "item_cnt_month",
    "avg_item_price",
    "revenue_month",
]

FORECAST_COLUMNS = [
    "id",
    "shop_id",
    "item_id",
    "item_category_id",
    "date_block_num",
    "prediction",
    "model_name",
]


@st.cache_data(ttl=300)
def read_gold_from_s3():
    df = wr.s3.read_parquet(GOLD_PATH, dataset=True)

    if "date_block_num" not in df.columns and "date_block_num" in df.index.names:
        df = df.reset_index()

    if "date_block_num" not in df.columns:
        raise KeyError(f"date_block_num no existe en Gold. Columnas disponibles: {df.columns.tolist()}")

    df["date_block_num"] = pd.to_numeric(df["date_block_num"], errors="coerce")

    available_cols = [col for col in GOLD_COLUMNS if col in df.columns]
    return df[available_cols].copy()


@st.cache_data(ttl=300)
def read_forecast_from_s3():
    df = wr.s3.read_parquet(FORECAST_PATH, dataset=True)

    if "date_block_num" not in df.columns and "date_block_num" in df.index.names:
        df = df.reset_index()

    if "date_block_num" in df.columns:
        df["date_block_num"] = pd.to_numeric(df["date_block_num"], errors="coerce")

    available_cols = [col for col in FORECAST_COLUMNS if col in df.columns]
    return df[available_cols].copy()


@st.cache_data(ttl=300)
def read_metrics_from_s3():
    try:
        return wr.s3.read_csv(METRICS_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def build_forecast_enriched(df_gold, df_forecast):
    catalog = (
        df_gold[
            [
                "shop_id",
                "item_id",
                "shop_name",
                "item_name",
                "item_category_name",
            ]
        ]
        .drop_duplicates(subset=["shop_id", "item_id"])
        .copy()
    )

    forecast_enriched = df_forecast.merge(
        catalog,
        on=["shop_id", "item_id"],
        how="left"
    )

    return forecast_enriched


@st.cache_data(ttl=300)
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def get_secret():
    client = boto3.client("secretsmanager", region_name=AWS_REGION)
    response = client.get_secret_value(SecretId=SECRET_NAME)
    return json.loads(response["SecretString"])


def get_connection():
    secret = get_secret()

    return psycopg2.connect(
        host=RDS_HOST,
        database=RDS_DATABASE,
        user=secret["username"],
        password=secret["password"],
        port=5432,
        sslmode="require"
    )


def save_feedback(texto):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (comentario)
            VALUES (%s)
            """,
            (texto,)
        )
        conn.commit()
        return True

    except Exception as e:
        st.error(f"Error al guardar feedback: {e}")
        return False

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def filter_by_months(df, months):
    if not months:
        return df.copy()
    return df[df["date_block_num"].isin(months)].copy()


def show_simple_feedback(section_name, key_prefix):
    st.divider()
    st.subheader("Feedback de negocio")

    comentario = st.text_area(
        "Comentario",
        key=f"{key_prefix}_comentario",
        placeholder="Escribe una observación de negocio sobre los datos o la proyección."
    )

    if st.button("Guardar feedback", key=f"{key_prefix}_guardar"):
        if comentario.strip() == "":
            st.warning("El comentario no puede estar vacío.")
        else:
            feedback_text = f"Sección: {section_name} | Comentario: {comentario}"
            if save_feedback(feedback_text):
                st.success("Feedback guardado en RDS usando Secrets Manager.")


def metric_card(col, label, value, prefix="", suffix=""):
    if pd.isna(value):
        col.metric(label, "N/D")
    else:
        col.metric(label, f"{prefix}{value:,.2f}{suffix}")


def top_table(df, group_col, n=5, ascending=False):
    summary = (
        df
        .groupby(group_col, as_index=False)
        .agg(
            ventas_totales=("item_cnt_month", "sum"),
            precio_promedio=("avg_item_price", "mean"),
            ingreso_total=("revenue_month", "sum")
        )
        .sort_values("ventas_totales", ascending=ascending)
        .head(n)
    )

    return summary


def add_line_chart_with_labels(df, x_col, y_col, color_col):
    base = alt.Chart(df).encode(
        x=alt.X(f"{x_col}:O", title="Mes"),
        y=alt.Y(f"{y_col}:Q", title=y_col),
        color=alt.Color(f"{color_col}:N", title="Categoría")
    )

    line = base.mark_line(point=True).encode(
        tooltip=[x_col, color_col, alt.Tooltip(y_col, format=",.2f")]
    )

    text = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-7,
        fontSize=10
    ).encode(
        text=alt.Text(f"{y_col}:Q", format=",.0f")
    )

    st.altair_chart(line + text, use_container_width=True)


def add_forecast_bar_chart(df):
    period_order = df["periodo"].tolist()

    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("periodo:N", title="Mes", sort=period_order),
        y=alt.Y("ventas:Q", title="Ventas"),
        color=alt.condition(
            alt.datum.tipo == "Proyección",
            alt.value("#ff7f0e"),
            alt.value("#1f77b4")
        ),
        tooltip=["periodo", "tipo", alt.Tooltip("ventas", format=",.2f")]
    )

    text = alt.Chart(df).mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=10
    ).encode(
        x=alt.X("periodo:N", sort=period_order),
        y=alt.Y("ventas:Q"),
        text=alt.Text("ventas:Q", format=",.0f")
    )

    st.altair_chart(bars + text, use_container_width=True)


try:
    df_gold = read_gold_from_s3()
    df_forecast = read_forecast_from_s3()
    df_metrics = read_metrics_from_s3()
    df_forecast_enriched = build_forecast_enriched(df_gold, df_forecast)

    if df_gold.empty:
        st.error("La tabla Gold está vacía.")
        st.stop()

    if df_forecast.empty:
        st.error("La tabla de forecast está vacía.")
        st.stop()

except Exception as e:
    st.error(f"No se pudieron cargar los datos desde AWS: {e}")
    st.stop()


months_available = sorted(df_gold["date_block_num"].dropna().unique())
shops_available = sorted(df_gold["shop_name"].dropna().unique())
categories_available = sorted(df_gold["item_category_name"].dropna().unique())


tab1, tab2, tab3 = st.tabs([
    "Trayectorias",
    "Proyecciones",
    "Descarga"
])


with tab1:
    st.header("Trayectorias")

    st.subheader("Estadísticas por periodo")
    selected_months_period = st.multiselect(
        "Periodo(s) - selecciona una opción o varias",
        options=months_available,
        default=[],
        key="period_months",
        help="Si no seleccionas periodos, se muestran todos los meses disponibles."
    )

    df_period = filter_by_months(df_gold, selected_months_period)

    top_shops = top_table(df_period, "shop_name", n=5, ascending=False)
    bottom_shops = top_table(df_period, "shop_name", n=5, ascending=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top 5 tiendas con mayores ventas")
        st.dataframe(
            top_shops.rename(columns={
                "shop_name": "Tienda",
                "ventas_totales": "Ventas totales",
                "precio_promedio": "Precio promedio",
                "ingreso_total": "Ingreso total",
            }),
            width="stretch",
            hide_index=True
        )

    with col2:
        st.write("Top 5 tiendas con menores ventas")
        st.dataframe(
            bottom_shops.rename(columns={
                "shop_name": "Tienda",
                "ventas_totales": "Ventas totales",
                "precio_promedio": "Precio promedio",
                "ingreso_total": "Ingreso total",
            }),
            width="stretch",
            hide_index=True
        )

    st.divider()

    st.subheader("Estadísticas por producto")

    col1, col2 = st.columns(2)

    with col1:
        selected_months_product = st.multiselect(
            "Periodo(s) - selecciona una opción o varias",
            options=months_available,
            default=[],
            key="product_months",
            help="Si no seleccionas periodos, se muestran todos los meses disponibles."
        )

    with col2:
        selected_shops_product = st.multiselect(
            "Tienda(s) - selecciona una opción o varias",
            options=shops_available,
            default=[],
            key="product_shops",
            help="Si no seleccionas tiendas, se muestran todas las tiendas disponibles."
        )

    df_product = filter_by_months(df_gold, selected_months_product)

    if selected_shops_product:
        df_product = df_product[df_product["shop_name"].isin(selected_shops_product)].copy()

    product_summary = (
        df_product
        .groupby(["item_id", "item_name", "item_category_name"], as_index=False)
        .agg(
            ventas_totales=("item_cnt_month", "sum"),
            precio_promedio=("avg_item_price", "mean"),
            ingreso_total=("revenue_month", "sum")
        )
    )

    top_products = product_summary.sort_values("ventas_totales", ascending=False).head(5)
    bottom_products = product_summary.sort_values("ventas_totales", ascending=True).head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top 5 productos con mayores ventas")
        st.dataframe(
            top_products.rename(columns={
                "item_id": "ID producto",
                "item_name": "Producto",
                "item_category_name": "Categoría",
                "ventas_totales": "Ventas totales",
                "precio_promedio": "Precio promedio",
                "ingreso_total": "Ingreso total",
            }),
            width="stretch",
            hide_index=True
        )

    with col2:
        st.write("Top 5 productos con menores ventas")
        st.dataframe(
            bottom_products.rename(columns={
                "item_id": "ID producto",
                "item_name": "Producto",
                "item_category_name": "Categoría",
                "ventas_totales": "Ventas totales",
                "precio_promedio": "Precio promedio",
                "ingreso_total": "Ingreso total",
            }),
            width="stretch",
            hide_index=True
        )

    st.divider()

    st.subheader("Detalle de estadísticas por tienda")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_shop_detail = st.selectbox(
            "Tienda - selecciona una opción o varias",
            options=shops_available,
            key="detail_shop"
        )

    df_detail_shop = df_gold[df_gold["shop_name"] == selected_shop_detail].copy()
    detail_categories_available = sorted(
        df_detail_shop["item_category_name"].dropna().unique()
    )

    with col2:
        selected_categories_detail = st.multiselect(
            "Tipo(s) de artículo - selecciona una opción o varias",
            options=detail_categories_available,
            default=detail_categories_available[:1],
            key="detail_categories"
        )

    with col3:
        selected_months_detail = st.multiselect(
            "Periodo(s) - selecciona una opción o varias",
            options=months_available,
            default=[],
            key="detail_months",
            help="Si no seleccionas periodos, se consideran todos los meses disponibles."
        )

    df_detail = df_detail_shop.copy()

    if selected_categories_detail:
        df_detail = df_detail[
            df_detail["item_category_name"].isin(selected_categories_detail)
        ].copy()

    df_detail_metrics = filter_by_months(df_detail, selected_months_detail)

    metrics_by_category = (
        df_detail_metrics
        .groupby("item_category_name", as_index=False)
        .agg(
            ventas_mensuales=("item_cnt_month", "sum"),
            precio_promedio=("avg_item_price", "mean"),
            ingreso_mensual=("revenue_month", "sum")
        )
        .sort_values("ventas_mensuales", ascending=False)
    )

    st.write("Métricas por tipo de artículo")
    st.dataframe(
        metrics_by_category.rename(columns={
            "item_category_name": "Tipo de artículo",
            "ventas_mensuales": "Ventas mensuales",
            "precio_promedio": "Precio promedio",
            "ingreso_mensual": "Ingreso mensual",
        }),
        width="stretch",
        hide_index=True
    )

    total_sales = df_detail_metrics["item_cnt_month"].sum()
    avg_price = df_detail_metrics["avg_item_price"].mean()
    total_revenue = df_detail_metrics["revenue_month"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Ventas mensuales", f"{total_sales:,.0f}")
    metric_card(col2, "Precio promedio", avg_price, prefix="$")
    metric_card(col3, "Ingreso mensual", total_revenue, prefix="$")

    st.subheader("Gráfica histórica")

    variable_options = {
        "Ventas mensuales": "item_cnt_month",
        "Precio promedio": "avg_item_price",
        "Ingreso mensual": "revenue_month"
    }

    selected_variable_label = st.selectbox(
        "Variable a graficar",
        list(variable_options.keys()),
        key="historical_variable"
    )

    selected_variable = variable_options[selected_variable_label]

    if selected_variable in ["item_cnt_month", "revenue_month"]:
        historical = (
            df_detail
            .groupby(["date_block_num", "item_category_name"], as_index=False)[selected_variable]
            .sum()
            .sort_values("date_block_num")
        )
    else:
        historical = (
            df_detail
            .groupby(["date_block_num", "item_category_name"], as_index=False)[selected_variable]
            .mean()
            .sort_values("date_block_num")
        )

    add_line_chart_with_labels(
        historical,
        x_col="date_block_num",
        y_col=selected_variable,
        color_col="item_category_name"
    )

    show_simple_feedback("Trayectorias", "feedback_trayectorias")


with tab2:
    st.header("Proyecciones")
    st.write(
        "El modelo pronostica ventas para el mes 34 por tienda y producto. "
        "La siguiente vista agrega las ventas históricas de la tienda seleccionada y "
        "contrasta el último periodo con la proyección."
    )

    selected_shop_forecast = st.selectbox(
        "Tienda",
        options=sorted(df_forecast_enriched["shop_name"].dropna().unique()),
        key="forecast_shop"
    )

    df_gold_shop = df_gold[df_gold["shop_name"] == selected_shop_forecast].copy()
    df_forecast_shop = df_forecast_enriched[
        df_forecast_enriched["shop_name"] == selected_shop_forecast
    ].copy()

    forecast_categories = sorted(df_forecast_shop["item_category_name"].dropna().unique())
    selected_forecast_categories = st.multiselect(
        "Categoría(s)",
        options=forecast_categories,
        default=[],
        key="forecast_categories",
        help="Si no seleccionas categorías, se consideran todas."
    )

    if selected_forecast_categories:
        df_gold_shop = df_gold_shop[
            df_gold_shop["item_category_name"].isin(selected_forecast_categories)
        ].copy()
        df_forecast_shop = df_forecast_shop[
            df_forecast_shop["item_category_name"].isin(selected_forecast_categories)
        ].copy()

    historical_shop = (
        df_gold_shop
        .groupby("date_block_num", as_index=False)
        .agg(ventas=("item_cnt_month", "sum"))
        .sort_values("date_block_num")
    )
    historical_shop["tipo"] = "Histórico"
    historical_shop["periodo"] = historical_shop["date_block_num"].astype(str)

    forecast_total = df_forecast_shop["prediction"].sum()
    forecast_month = int(df_forecast_shop["date_block_num"].dropna().max())

    forecast_row = pd.DataFrame({
        "date_block_num": [forecast_month],
        "ventas": [forecast_total],
        "tipo": ["Proyección"],
        "periodo": [f"{forecast_month} (forecast)"]
    })

    projection_table = pd.concat(
        [historical_shop[["date_block_num", "ventas", "tipo", "periodo"]], forecast_row],
        ignore_index=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tienda seleccionada", selected_shop_forecast)
    col2.metric("Pronóstico mes 34", f"{forecast_total:,.0f}")
    col3.metric("Productos proyectados", f"{df_forecast_shop['item_id'].nunique():,.0f}")
    col4.metric("Modelo", df_forecast_shop["model_name"].dropna().iloc[0] if not df_forecast_shop.empty else "N/D")

    st.subheader("Histórico y proyección")

    projection_display = projection_table[["periodo", "ventas", "tipo"]].copy()

    st.dataframe(
        projection_display.style.apply(
            lambda row: [
                "background-color: #fff3cd; color: black; font-weight: bold"
                if row["tipo"] == "Proyección" else ""
                for _ in row
            ],
            axis=1
        ),
        width="stretch",
        hide_index=True
    )

    projection_chart = projection_table[["date_block_num", "ventas", "tipo"]].copy()
    projection_chart["date_block_num"] = projection_chart["date_block_num"].astype(int)
    projection_chart = projection_chart.sort_values("date_block_num").reset_index(drop=True)
    projection_chart["periodo"] = projection_chart["date_block_num"].astype(str)

    projection_chart.loc[
        projection_chart["tipo"] == "Proyección",
        "periodo"
    ] = projection_chart["date_block_num"].astype(str) + " (forecast)"

    add_forecast_bar_chart(projection_chart[["periodo", "ventas", "tipo"]])    
    
    st.subheader("Métricas del modelo")

    if df_metrics.empty:
        st.info("No se encontraron métricas del modelo en S3.")
    else:
        st.dataframe(df_metrics, width="stretch", hide_index=True)

        numeric_metrics = df_metrics.select_dtypes(include="number")
        if not numeric_metrics.empty:
            metric_cols = st.columns(min(4, len(numeric_metrics.columns)))
            for idx, metric_name in enumerate(numeric_metrics.columns[:4]):
                metric_card(
                    metric_cols[idx],
                    metric_name,
                    numeric_metrics[metric_name].iloc[0]
                )

        st.write(
            "Estas métricas comparan el desempeño del modelo XGBoost contra un baseline naive. "
            "MAE y RMSE más bajos indican menor error promedio de predicción."
        )

    show_simple_feedback("Proyecciones", "feedback_proyecciones")


with tab3:
    st.header("Descarga")

    st.write("Descarga la tabla Gold completa para análisis batch o validación externa.")

    csv_gold = convert_df_to_csv(df_gold)

    st.download_button(
        label="Descargar tabla Gold completa",
        data=csv_gold,
        file_name="gold_modeling_sales.csv",
        mime="text/csv"
    )

    st.divider()

    st.write("También puedes descargar el forecast enriquecido completo con nombres de tienda, producto y categoría.")

    forecast_download_cols = [
        "id",
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "item_category_name",
        "date_block_num",
        "prediction",
        "model_name"
    ]

    csv_forecast = convert_df_to_csv(df_forecast_enriched[forecast_download_cols])

    st.download_button(
        label="Descargar forecast enriquecido completo",
        data=csv_forecast,
        file_name="forecast_xgboost_enriquecido.csv",
        mime="text/csv"
    )
