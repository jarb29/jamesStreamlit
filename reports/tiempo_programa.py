from util_plot import *
from util_functions import *
import streamlit as st
original_df = pd.read_csv('data/saved_df.csv')

messages = ["Open File: C:", 'Total machining', 'total', 'Layer: 1']

new_message_df = get_surrounding_rows(original_df, messages, 'Message', 0)


filtered_dataframe = filter_open_file_messages(new_message_df, 6)


filtered_dataframe = drop_open_file_duplicates(filtered_dataframe)

total_machining_df = total_machining_per_program(filtered_dataframe)


fig = create_barplot(
    df=total_machining_df,
    x_col='program',
    y_col='total_machining',
    x_title="Nombre del Programa", y_title="Tiempo de Corte (min)"
)

st.plotly_chart(fig)
