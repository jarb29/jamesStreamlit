import pandas as pd

from util_plot import *
from util_functions import *
import streamlit as st
original_df = pd.read_excel('data/tiempo_final.xlsx')

tiempo_total_df = group_and_sum(original_df, 'Espesor', 'Programas cortados')

# apply transformations to the DataFrame
df_reset = transform_data(original_df)

# merge the transformed DataFrame with the grouped and summed DataFrame on 'Espesor' column
df_reset = pd.merge(tiempo_total_df , df_reset, on='Espesor')

print(df_reset)
columns = ['Longitude Corte (m)', 'Time (min)', 'Velocidad (m/min)', 'Programas cortados']

fig = sunburst_plot(df_reset, columns)
st.plotly_chart(fig)