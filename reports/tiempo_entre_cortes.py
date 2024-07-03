from util_plot import *
from util_functions import *
import streamlit as st
original_df = pd.read_csv('data/saved_df.csv')
df_tiempo_carga, df2 = time_between_placas(original_df, ["Layer: 1", "Total machining"])

grouped_dgf, months, years = extract_month_year(df2)

# Let the user select the month and year
selected_month = st.sidebar.selectbox('Select a month', months)
selected_year = st.sidebar.selectbox('Select a year', years)

# Filter based on the user selection
filtered_df = df2[(df2['Month'] == selected_month) & (df2['Year'] == selected_year)]

fig = plot_distribution(filtered_df, 'Timestamp_Diff', min_count=5)
st.plotly_chart(fig)