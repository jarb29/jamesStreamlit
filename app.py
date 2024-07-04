import streamlit as st
import altair as alt
from util_functions import *
from util_plot import *


original_df = pd.read_csv('data/saved_df.csv')

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

original_df_tiempo = pd.read_excel('data/tiempo_final.xlsx')
tiempo_total_df = group_and_sum(original_df_tiempo, 'timestamp', 'Espesor', 'Programas cortados')
df_reset = transform_data(original_df_tiempo, 'timestamp')
df_reset = pd.merge(tiempo_total_df , df_reset, on='Espesor')
df_reset = df_reset.drop(columns='Date_y')  # drop one of the month columns
df_reset = df_reset.rename(columns={'Date_x': 'Date'})  # rename 'month_x' to 'Month'
df_5, m5, y5 = extract_month_year(df_reset)


# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
#
# def login():
#     if st.button("Log in"):
#         st.session_state.logged_in = True
#         st.rerun()
#
# def logout():
#     if st.button("Log out"):
#         st.session_state.logged_in = False
#         st.rerun()

# login_page = st.Page(login, title="Log in", icon=":material/login:")
# logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
st.set_page_config(
    page_title="Kupfer Nave1/Laser Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

with st.sidebar:
    st.title("📅 Kupfer Nave1/Laser Dashboard")
    selected_month = st.sidebar.selectbox('Select a month', months)
    selected_year = st.sidebar.selectbox('Select a year', years)

# dashboard = st.Page(
#     "reports/tiempo_inicio.py", title="Tiempo de inicio", icon=":material/dashboard:", default=True
# )
# bugs = st.Page("reports/tiempo_entre_cortes.py", title="Tiempo entre inicio de cortes", icon=":material/bug_report:")
# alerts = st.Page(
#     "reports/tiempo_diario.py", title="Tiempo/Dia", icon=":material/notification_important:"
# )
# tiempo = st.Page(
#     "reports/tiempo_programa.py", title="Tiempo/Programa", icon=":material/notification_important:"
# )
# reporte = st.Page(
#     "reports/reporte.py", title="Reporte", icon=":material/history:"
# )
# search = st.Page("tools/search.py", title="Search", icon=":material/search:")
# history = st.Page("tools/history.py", title="History", icon=":material/history:")
#
# # if st.session_state.logged_in:
# pg = st.navigation(
#         {
#             # "Account": [logout_page],
#             "Informe": [dashboard, bugs, alerts, tiempo, reporte],
#             # "Tools": [search, history],
#         }
#     )
# else:
#     pg = st.navigation([login_page])

#######################
# Dashboard Main Panel
col = st.columns((1, 4, 4), gap='medium')

with col[0]:
    st.markdown('#### Gains/Losses')

    # df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_year)
    #
    # if selected_year > 2010:
    #     first_state_name = df_population_difference_sorted.states.iloc[0]
    #     first_state_population = format_number(df_population_difference_sorted.population.iloc[0])
    #     first_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[0])
    # else:
    #     first_state_name = '-'
    #     first_state_population = '-'
    #     first_state_delta = ''
    # st.metric(label=first_state_name, value=first_state_population, delta=first_state_delta)
    #
    # if selected_year > 2010:
    #     last_state_name = df_population_difference_sorted.states.iloc[-1]
    #     last_state_population = format_number(df_population_difference_sorted.population.iloc[-1])
    #     last_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[-1])
    # else:
    #     last_state_name = '-'
    #     last_state_population = '-'
    #     last_state_delta = ''
    # st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)

    st.markdown('#### States Migration')

    # if selected_year > 2010:
    #     # Filter states with population difference > 50000
    #     # df_greater_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference_absolute > 50000]
    #     df_greater_50000 = df_population_difference_sorted[
    #         df_population_difference_sorted.population_difference > 50000]
    #     df_less_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference < -50000]
    #
    #     # % of States with population difference > 50000
    #     states_migration_greater = round(
    #         (len(df_greater_50000) / df_population_difference_sorted.states.nunique()) * 100)
    #     states_migration_less = round((len(df_less_50000) / df_population_difference_sorted.states.nunique()) * 100)
    #     donut_chart_greater = make_donut(states_migration_greater, 'Inbound Migration', 'green')
    #     donut_chart_less = make_donut(states_migration_less, 'Outbound Migration', 'red')
    # else:
    #     states_migration_greater = 0
    #     states_migration_less = 0
    #     donut_chart_greater = make_donut(states_migration_greater, 'Inbound Migration', 'green')
    #     donut_chart_less = make_donut(states_migration_less, 'Outbound Migration', 'red')

    # migrations_col = st.columns((0.2, 1, 0.2))
    # with migrations_col[1]:
    #     st.write('Inbound')
    #     st.altair_chart(donut_chart_greater)
    #     st.write('Outbound')
    #     st.altair_chart(donut_chart_less)

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

# pg.run()