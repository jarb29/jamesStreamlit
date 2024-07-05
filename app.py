import pandas as pd
import streamlit as st
import altair as alt
from util_functions import *
from util_plot import *


original_df = pd.read_csv('data/saved_df.csv')
corte_sap = pd.read_excel('data/CORTE_SAP.xlsx')
original_df_tiempo = pd.read_excel('data/tiempo_final.xlsx')

# first plot tiempo_diario
grouped_df = group_by_date(original_df)
df_1, months, years = extract_month_year(grouped_df)

###########################################
df_tiempo_, df_2 = time_between_placas(original_df, ["Layer: 1", "Total machining"])
df_2, m2, y2 = extract_month_year(df_2)

############################################
df_3 = first_occurrence_per_date(original_df, 'Message', 'Layer: 1')
df_3, m3, y3 = extract_month_year(df_3)

###########################################
messages = ["Open File: C:", 'Total machining', 'total', 'Layer: 1']
new_message_df = get_surrounding_rows(original_df, messages, 'Message', 0)
filtered_dataframe = filter_open_file_messages(new_message_df, 6)
filtered_dataframe = drop_open_file_duplicates(filtered_dataframe)

df_4 = total_machining_per_program(filtered_dataframe)
df_4["Date"] = df_4['timestamp']
df_4, m4, y4 = extract_month_year(df_4)
df_4 = df_4.sort_values(by="total_machining")

##############################################


tiempo_total_df = group_and_sum(original_df_tiempo, 'timestamp', 'Espesor', 'Programas cortados')
df_reset = transform_data(original_df_tiempo, 'timestamp')
df_reset = pd.merge(tiempo_total_df , df_reset, on='Espesor')
df_reset = df_reset.drop(columns='Date_y')  # drop one of the month columns
df_reset = df_reset.rename(columns={'Date_x': 'Date'})  # rename 'month_x' to 'Month'
df_5, m5, y5 = extract_month_year(df_reset)

#################################################

corte_sap['Date'] = pd.to_datetime(corte_sap['Fec.Produc'], format='%d.%m.%Y')
df_6, m6, y6 = extract_month_year(corte_sap)





st.set_page_config(
    page_title="Kupfer Nave1/Laser Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

with st.sidebar:
    st.sidebar.image("data/logo.png", use_column_width=True)
    st.title("📅 Nave1/Laser Dashboard")
    selected_month = st.sidebar.selectbox('Select a month', months)
    selected_year = st.sidebar.selectbox('Select a year', years)





#######################
# Dashboard Main Panel
col = st.columns((2, 4, 4), gap='medium')

with col[0]:

    ###### Logitud y tiempo

    filtered_time = df_1[(df_1['Month'] == selected_month) & (df_1['Year'] == selected_year)]
    filtered_time_1 = df_1[(df_1['Month'] == (selected_month-1)) & (df_1['Year'] == selected_year)]
    time = round(sum(filtered_time['Time']), 2)
    time_1 = round(sum(filtered_time_1['Time']),2 )
    delta_time = round(time - time_1,2)
    filtered_df6 = df_5[(df_5['Month'] == selected_month) & (df_5['Year'] == selected_year)]
    filtered_df6_1 = df_5[(df_5['Month'] == (selected_month-1)) & (df_5['Year'] == selected_year)]

    avg_espesor = round(weighted_average_espesor(filtered_df6))
    avg_espesor_1 = weighted_average_espesor(filtered_df6_1)
    delta_espesor = round(avg_espesor-avg_espesor_1)
    filtered_sap_df6 = df_6[(df_6['Month'] == selected_month) & (df_6['Year'] == selected_year)]
    filtered_sap_df6_1 = df_6[(df_6['Month'] == (selected_month-1)) & (df_6['Year'] == selected_year)]



    if len(filtered_sap_df6) > 0:

        st.markdown('### Produccion')
        kg = round(sum(filtered_sap_df6['Kg Producc']),2)
        piezas  = round(sum(filtered_sap_df6['Tot.cant.p']), 2)
        kg_1= round(sum(filtered_sap_df6_1['Kg Producc']),2)
        piezas_1  = round(sum(filtered_sap_df6_1['Tot.cant.p']), 2)
        delta_kg = round(kg - kg_1, 2)
        delta_piezas = round(piezas-piezas_1, 2)
        st.metric(label='Kg. Cortados', value=kg, delta=delta_kg)
        st.markdown("<hr style='margin:25px 0px;width:50%;border-color:lightgray'>", unsafe_allow_html=True)
        st.metric(label='Cantidad de Piezas', value=piezas, delta=delta_piezas)
        st.markdown("<hr style='margin:25px 0px;width:50%;border-color:lightgray'>", unsafe_allow_html=True)
        st.metric(label="Espesor promedio", value=avg_espesor, delta=delta_espesor)

    st.markdown('- - - - - - - - - - ')
    st.markdown('## Corte')
    st.metric(label='Tiempo (min)', value=time, delta=delta_time)
    st.markdown("<hr style='margin:25px 0px;width:50%;border-color:lightgray'>", unsafe_allow_html=True)
    if len(filtered_df6) > 0:
        longitud_corte = round(sum(filtered_df6['Longitude Corte (m)']), 2)
        longitud_corte_1 = round(sum(filtered_df6_1['Longitude Corte (m)']), 2)
        delta_logitud = round(longitud_corte - longitud_corte_1, 2)

            # time_corte = round(sum(filtered_df6['Time (min)']), 2)
            # time_corte_1 = round(sum(filtered_df_1['Time (min)']), 2)

        st.metric(label='Logitud (m)', value=longitud_corte, delta=delta_logitud)
        # time_corte = round(sum(filtered_df6['Time (min)']), 2)
        # time_corte_1 = round(sum(filtered_df_1['Time (min)']), 2)








with col[1]:
    filtered_df4 = df_4[(df_4['Month'] == selected_month) & (df_4['Year'] == selected_year)]
    if len(filtered_df4) > 0:
        st.markdown('#### Tiempo por Programa')

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

    st.markdown('#### Tiempo Laser ON por dia')

    filtered_df_1 = df_1[(df_1['Month'] == selected_month) & (df_1['Year'] == selected_year)]

    fig1 = plot_daily_time(filtered_df_1, selected_month)

    st.plotly_chart(fig1)
    st.markdown('#### Tiempo entre Cortes')
    filtered_df_2 = df_2[(df_2['Month'] == selected_month) & (df_2['Year'] == selected_year)]

    fig2 = plot_distribution(filtered_df_2, 'Timestamp_Diff', min_count=5)
    st.plotly_chart(fig2)

with col[2]:
    st.markdown('#### Tiempo de inicio del Laser')

    filtered_df3 = df_3[(df_3['Month'] == selected_month) & (df_3['Year'] == selected_year)]

    fig3 = plot_time(filtered_df3)
    st.plotly_chart(fig3)



    filtered_df5= df_5[(df_5['Month'] == selected_month) & (df_5['Year'] == selected_year)]

    if len(filtered_df5)> 0:
        st.markdown('#### Resumen')
        columns_names = ['Longitude Corte (m)', 'Time (min)', 'Velocidad (m/min)', 'Programas cortados']

        fig5 = sunburst_plot(filtered_df5, columns_names)
        st.plotly_chart(fig5)

    with st.expander('Sobre', expanded=True):
        st.write('''
            - Laser: [Senfeng Company](https://www.senfenglaser.es/?gad_source=1&gclid=CjwKCAjwkJm0BhBxEiwAwT1AXJKYI8MvgvXsXH2OaMG0nQnX717QGvxNLJUrxU2erMDDz9jpYBYEUhoCmMUQAvD_BwE).
            - :orange[**Serie**]: 12000
            - :orange[**Ubicacion**]: Coquimbo, Colina, Región Metropolitana
            ''')

