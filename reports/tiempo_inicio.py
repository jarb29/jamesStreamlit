from util_plot import *
from util_functions import *
import streamlit as st

original_df = pd.read_csv('data/saved_df.csv')

df_first_occurrence = first_occurrence_per_date(original_df, 'Message', 'Layer: 1')


grouped_dgf, months, years = extract_month_year(df_first_occurrence )

# Let the user select the month and year
selected_month = st.sidebar.selectbox('Select a month', months)
selected_year = st.sidebar.selectbox('Select a year', years)

# Filter based on the user selection
filtered_df = grouped_dgf[(grouped_dgf['Month'] == selected_month) & (grouped_dgf['Year'] == selected_year)]

fig = plot_time(filtered_df)
st.plotly_chart(fig)