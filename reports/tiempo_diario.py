from util_functions import *
from util_plot import *
import streamlit as st
import plotly.graph_objects as go

original_df = pd.read_csv('data/saved_df.csv')



grouped_df = group_by_date(original_df)


grouped_dgf, months, years = extract_month_year(grouped_df)

# Let the user select the month and year
selected_month = st.sidebar.selectbox('Select a month', months)
selected_year = st.sidebar.selectbox('Select a year', years)

# Filter based on the user selection
filtered_df = grouped_dgf[(grouped_dgf['Month'] == selected_month) & (grouped_dgf['Year'] == selected_year)]


fig = plot_daily_time(filtered_df, selected_month)

st.plotly_chart(fig)

