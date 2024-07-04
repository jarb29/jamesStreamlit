import pandas as pd

from util_plot import *
from util_functions import *
import streamlit as st
original_df = pd.read_excel('data/tiempo_final.xlsx')
tiempo_total_df = group_and_sum(original_df, 'timestamp', 'Espesor', 'Programas cortados')
# print(tiempo_total_df, 'tiempo_total_df')
# apply transformations to the DataFrame
df_reset = transform_data(original_df, 'timestamp')
# merge the transformed DataFrame with the grouped and summed DataFrame on 'Espesor' column
df_reset = pd.merge(tiempo_total_df , df_reset, on='Espesor')
df_reset = df_reset.drop(columns='Date_y')  # drop one of the month columns
# df_reset['month_x'] = pd.to_datetime(df_reset['month_x'], format='%B').dt.month  # convert month name to number
df_reset = df_reset.rename(columns={'Date_x': 'Date'})  # rename 'month_x' to 'Month'
grouped_dgf, months, years = extract_month_year(df_reset)

# Let the user select the month and year
selected_month = st.sidebar.selectbox('Select a month', months)
selected_year = st.sidebar.selectbox('Select a year', years)

# Filter based on the user selection
filtered_df = grouped_dgf[(grouped_dgf['Month'] == selected_month) & (grouped_dgf['Year'] == selected_year)]


columns = ['Longitude Corte (m)', 'Time (min)', 'Velocidad (m/min)', 'Programas cortados']

fig = sunburst_plot(filtered_df, columns)
st.plotly_chart(fig)