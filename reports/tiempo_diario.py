from util_functions import *
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

original_df = pd.read_csv('data/saved_df.csv')



grouped_df = group_by_date(original_df)


grouped_dgf, months, years = extract_month_year(grouped_df)

# Let the user select the month and year
selected_month = st.sidebar.selectbox('Select a month', months)
selected_year = st.sidebar.selectbox('Select a year', years)

# Filter based on the user selection
filtered_df = grouped_df[(grouped_df['Month'] == selected_month) & (grouped_df['Year'] == selected_year)]


# Assume that 'filtered_df' is your DataFrame
fig = px.line(filtered_df, y="Time", x="Date")

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Time"],
        mode='markers',
        marker=dict(
            color='red',
        ),
        text=filtered_df["Time"].round(2),
        hovertemplate='%{text}',
    )
)

st.plotly_chart(fig)

