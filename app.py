import streamlit as st
import altair as alt
import pandas as pd
import plotly.express as px
from util_functions import *
from util_plot import *

# --- Data Loading ---
corte_sap = pd.read_excel('data/CORTE_SAP.xlsx')
corte_sap = strip_column_names(corte_sap)
original_df_tiempo = pd.read_excel('data/tiempo_final.xlsx')
months, years, cm, cy = get_months_and_years_since("01/04/2024")

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Dashboard/Laser",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded")

# --- Custom Theme for Better Colors and Styling ---
st.markdown("""
<style>
/* General Styling */
body, h1, h2, h3, h4, .stMetric label, .stMetric div[data-testid="stMetricValue"], .stDataFrame, p {
    color: #000000 !important; 
    font-family: 'Arial', sans-serif;
}
body {
    background-color: #f4f4f4;
}
.stApp {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

/* Header Styling */
.stHeader {
    background-color: #e9ecef;  
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 15px;
}

/* Metric Styling */
.stMetric {
    border: 1px solid #e0e0e0;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
    font-size: 24px !important; 
    font-weight: bold !important; 
}

/* Plot Styling */
.plotly-graph-svg {
    background-color: #ffffff;
}
.plotly-graph-svg text { 
    fill: #000000 !important; 
} 

/* Expander Styling */
.stExpanderHeader {
    background-color: #e9ecef;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    border: 1px solid #cccccc; 
}
.stExpanderContent a { 
    color: orange !important;
}
.stExpanderContent p strong:nth-child(1),
.stExpanderContent p strong:nth-child(3) 
{
    color: red !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.sidebar.image("data/logo.png", use_column_width=True)
    st.title("📅 Nave1/Laser Dashboard")
    default_month_index = months.index(cm) - 1
    default_years_index = years.index(cy)
    selected_month = st.sidebar.selectbox('Select a month', months, index=default_month_index)
    selected_year = st.sidebar.selectbox('Select a year', years, index=default_years_index)

# --- Data Processing ---
original_df = pd.read_csv(f'data/saved_df_{selected_year}_{selected_month}.csv')
var1 = selected_month - 1
if selected_month == 4 and selected_year == 2024:
    var1 = selected_month
original_df_1 = pd.read_csv(f'data/saved_df_{selected_year}_{var1}.csv')

# Data processing for plots
df_1 = group_by_date(original_df)
df_1 = extract_month_year(df_1)
original_df_1 = group_by_date(original_df_1)
original_df_1 = extract_month_year(original_df_1)

df_tiempo_, df_2 = time_between_placas(original_df, ["Layer: 1", "Total machining"])
df_2 = extract_month_year(df_2)

df_3 = first_occurrence_per_date(original_df, 'Message', 'Layer: 1')
df_3 = extract_month_year(df_3)

messages = ["Open File: C:", 'Total machining', 'total', 'Layer: 1']
new_message_df = get_surrounding_rows(original_df, messages, 'Message', 0)
filtered_dataframe = filter_open_file_messages(new_message_df, selected_month)
filtered_dataframe = drop_open_file_duplicates(filtered_dataframe)

df_4 = total_machining_per_program(filtered_dataframe)
df_4["Date"] = df_4['timestamp']
df_4 = df_4.sort_values(by="total_machining")
df_4 = extract_month_year(df_4)

filtered_time = df_1[(df_1['Month'] == selected_month) & (df_1['Year'] == selected_year)]
filtered_time_1 = original_df_1[(original_df_1['Month'] == (selected_month - 1)) & (original_df_1['Year'] == selected_year)]
time = round(sum(filtered_time['Time'] / 60), 2)
time_1 = round(sum(filtered_time_1['Time'] / 60), 2)
delta_time = round(time - time_1, 2)

tiempo_total_df = group_and_sum(original_df_tiempo, 'timestamp', 'Espesor', 'Programas cortados')
df_reset = transform_data(original_df_tiempo, 'timestamp')
df_5 = pd.merge(tiempo_total_df, df_reset, on=['Espesor', 'Date'])
df_5 = extract_month_year(df_5)
filtered_df6 = df_5[(df_5['Month'] == selected_month) & (df_5['Year'] == selected_year)]
filtered_df6_1 = df_5[(df_5['Month'] == (selected_month - 1)) & (df_5['Year'] == selected_year)]

corte_sap['Date'] = pd.to_datetime(corte_sap['Fec.Produc'], format='%d.%m.%Y')
avg_espesor = round(weighted_average_espesor(filtered_df6))
avg_espesor_1 = weighted_average_espesor(filtered_df6_1)
delta_espesor = round(avg_espesor - avg_espesor_1)
corte_sap = extract_month_year(corte_sap)
filtered_sap_df6 = corte_sap[(corte_sap['Month'] == selected_month) & (corte_sap['Year'] == selected_year)]
filtered_sap_df6_1 = corte_sap[(corte_sap['Month'] == (selected_month - 1)) & (corte_sap['Year'] == selected_year)]

# --- Dashboard Main Panel ---
st.markdown("<div class='stHeader'><h1>Kupfer Nave1/Laser Dashboard</h1></div>", unsafe_allow_html=True)

# --- Column Layout ---
col1, col2 = st.columns((1, 3))

# --- Produccion & Corte Metrics Section ---
with col1:
    if len(filtered_sap_df6) > 0:
        st.markdown('<h3>Produccion</h3>', unsafe_allow_html=True)
        kg = round(filtered_sap_df6['Kg Producc'].sum(skipna=True), 2)
        piezas  = round(filtered_sap_df6['Tot.cant.p'].sum(skipna=True), 2)
        kg_1= round(filtered_sap_df6_1['Kg Producc'].sum(skipna=True), 2)
        piezas_1  = float(round(filtered_sap_df6_1['Tot.cant.p'].sum(skipna=True), 2))

        delta_kg = round(kg - kg_1, 2)
        delta_piezas = round(piezas-piezas_1, 2)
        st.metric(label='Kg. Cortados', value=kg, delta=delta_kg)
        st.markdown("<hr style='margin:25px 0px;width:50%;border-color:lightgray'>", unsafe_allow_html=True)
        st.metric(label='Cantidad de Piezas', value=piezas, delta=delta_piezas)
        st.markdown("<hr style='margin:25px 0px;width:50%;border-color:lightgray'>", unsafe_allow_html=True)
        st.metric(label="Espesor promedio", value=avg_espesor, delta=delta_espesor)

    st.markdown('- - - - - - - - - - ')
    st.markdown('<h2>Corte</h2>', unsafe_allow_html=True)
    st.metric(label='Tiempo (Hr)', value=time, delta=delta_time)
    st.markdown("<hr style='margin:25px 0px;width:50%;border-color:lightgray'>", unsafe_allow_html=True)
    if len(filtered_df6) > 0:
        longitud_corte = round(filtered_df6['Longitude Corte (m)'].sum(skipna=True), 2)
        longitud_corte_1 = round(filtered_df6_1['Longitude Corte (m)'].sum(skipna=True), 2)
        delta_logitud = round(longitud_corte - longitud_corte_1, 2)
        st.metric(label='Logitud (m)', value=longitud_corte, delta=delta_logitud)

    # --- Tiempo por Programa DataFrame ---
    filtered_df4 = df_4[(df_4['Month'] == selected_month) & (df_4['Year'] == selected_year)]
    if len(filtered_df4) > 0:
        st.markdown('<h4>Tiempo por Programa</h4>', unsafe_allow_html=True)
        st.dataframe(filtered_df4,
                     column_order=("program", "total_machining"),
                     hide_index=True,
                     width=None,
                     column_config={
                         "program": st.column_config.TextColumn(
                             "program",
                         ),
                         "total_machining": st.column_config.ProgressColumn(
                             "total_machining",
                             format="%f",
                             min_value=0,
                             max_value=max(df_4.total_machining),
                         )}
                     )

# --- Plots Section ---
with col2:
    # Plot 1: Tiempo Laser ON por dia
    st.markdown('<h4>Tiempo Laser ON por dia</h4>', unsafe_allow_html=True)
    filtered_df_1 = df_1[(df_1['Month'] == selected_month) & (df_1['Year'] == selected_year)]
    fig1 = plot_daily_time(filtered_df_1, selected_month)
    st.plotly_chart(fig1, use_container_width=True, height=400)

    # Plot 2: Tiempo entre Cortes
    st.markdown('<h4>Tiempo entre Cortes</h4>', unsafe_allow_html=True)
    filtered_df_2 = df_2[(df_2['Month'] == selected_month) & (df_2['Year'] == selected_year)]
    fig2 = plot_distribution(filtered_df_2, 'Timestamp_Diff', min_count=5)
    st.plotly_chart(fig2, use_container_width=True, height=400)

    # Plot 3: Tiempo de inicio del Laser
    st.markdown('<h4>Tiempo de inicio del Laser</h4>', unsafe_allow_html=True)
    filtered_df3 = df_3[(df_3['Month'] == selected_month) & (df_3['Year'] == selected_year)]
    fig3 = plot_time(filtered_df3)
    st.plotly_chart(fig3, use_container_width=True, height=400)

    # Plot 4: Resumen
    filtered_df5 = df_5[(df_5['Month'] == selected_month) & (df_5['Year'] == selected_year)]
    if len(filtered_df5) > 0:
        st.markdown('<h4>Resumen</h4>', unsafe_allow_html=True)
        columns_names = ['Longitude Corte (m)', 'Time (min)', 'Velocidad (m/min)', 'Programas cortados']
        fig5 = sunburst_plot(filtered_df5, columns_names)
        st.plotly_chart(fig5, use_container_width=True, height=400)

# --- Sobre Section ---
with st.expander('Sobre', expanded=True):
    st.write('''
        - Laser: [Senfeng Company](https://www.senfenglaser.es/?gad_source=1&gclid=CjwKCAjwkJm0BhBxEiwAwT1AXJKYI8MvgvXsXH2OaMG0nQnX717QGvxNLJUrxU2erMDDz9jpYBYEUhoCmMUQAvD_BwE).
        - :orange[**Serie**]: 12000
        - :orange[**Ubicacion**]: Coquimbo, Colina, Región Metropolitana
        ''')