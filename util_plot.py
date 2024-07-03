import math
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def plot_per_month(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set Date as index for easy resampling
    df.set_index('Date', inplace=True)

    # Get the unique months in the data
    months = df.index.to_period('M').unique()

    for month in months:
        # Extract rows for this month
        monthly_data = df[df.index.to_period('M') == month]

        # Compute the monthly average
        monthly_average = monthly_data['Planchas Cortadas'].mean()

        # Plot the daily values for this month
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=monthly_data.index.day,
                     y=monthly_data['Planchas Cortadas'],
                     marker='o')

        # Plus annotate each point with its y value
        for line in plt.gca().lines:
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                plt.text(x, y, f'{y:.2f}',
                         color='red',
                         fontsize=7,
                         ha='center',
                         va='bottom')

        # Plot the average line
        plt.axhline(y=monthly_average, color='r', linestyle='-', label=f'Average: {monthly_average:.2f}')

        plt.xlabel('Dia del Mes')
        plt.ylabel('Planchas Cortadas')
        plt.title(f'Planchas Cortadas {month.strftime("%B %Y")}')
        plt.legend(title='')
        plt.grid(True)
        plt.savefig(f'plots/Planchas Cortadas {month.strftime("%B %Y")}')
        plt.show()



def plot_daily(df, date_col, val_col):
    import matplotlib.pyplot as plt

    # Set 'date_col' as the index
    df = df.set_index(date_col).copy()
    df[val_col] = df[val_col] / 60  # Converting time to hours

    # Get unique months in the DataFrame
    months = df.index.month.unique()

    for month in months:
        # Filter df for each month
        df_month = df[df.index.month == month]

        # Calculate average
        average = df_month[val_col].mean()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the DataFrame
        df_month[val_col].plot(ax=ax, label=val_col, marker='o')  # Added marker='o' to put dots on each data point.

        # Annotate each point with its y value
        for x, y in zip(df_month.index, df_month[val_col]):
            ax.annotate('{:.2f}'.format(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')

        # Adding average line and creating a Line2D object for the legend
        ax.axhline(y=average, color='r', linestyle='-')
        ax.plot([], [], 'r-', label=f'Average: {average:.2f}')

        plt.title(f'Laser ON mes: {month}', fontsize=15)
        plt.xlabel('Dia del Mes', fontsize=13)  # Changed 'Day' to 'Tiempo en el mes'
        plt.ylabel('Tiempo/Horas', fontsize=13)

        ax.legend(loc='best')
        plt.savefig(f'plots/Laser ON mes {month}')
        plt.show()


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_distribution(df, column, min_count=5):
    # Create a new column 'Month' from 'Date'
    df['Month'] = pd.DatetimeIndex(df['Date']).month

    # Get list of months in the DataFrame
    months = df['Month'].unique()

    # Create subplots, where each subplot is for a month
    fig = make_subplots(rows=len(months), cols=1)

    for i, month in enumerate(months, start=1):
        # Filter df for each month and count
        df_month = df[df['Month'] == month]
        counts = df_month[column].value_counts()
        values_to_keep = counts[(counts >= min_count) & (counts != 0)].index
        data_filtered = df_month[df_month[column].isin(values_to_keep)]

        # Draw histogram
        hist_data = go.Histogram(
            x=data_filtered[column],
            nbinsx=30,
            name=f'Mes: {month}',
            marker=dict(
                line=dict(
                    width=1  # Gives space between bars.
                )
            ),
        )
        fig.add_trace(hist_data, row=i, col=1)

        # Calculate the mean (average)
        mean_value = data_filtered[column].mean()

        # Add vertical line for the average
        fig.add_shape(
            type="line",
            x0=mean_value,
            y0=0,
            x1=mean_value,
            y1=1,
            yref='paper',
            line=dict(
                color="Red",
                width=3   # Makes the line thicker.
            ),
            row=i,
            col=1
        )

        # Update yaxis properties
        fig.update_yaxes(title_text="# repeticiones", row=i, col=1)

        # Update xaxis properties
        fig.update_xaxes(title_text="Tiempo entre cortes", row=i, col=1)

        # Create a dummy scatter plot to add the mean value to the legend
        # Note: The point of this scatter plot is invisible
        trace = go.Scatter(
            x=[mean_value],
            y=[data_filtered[column].max()],
            mode='markers',
            marker=dict(
                size=1,
                color='rgba(0, 0, 0, 0)'  # marker is invisible
            ),
            showlegend=True,
            name=f'Promedio: {mean_value:.2f}, minutos',  # this will appear in the legend
        )
        fig.add_trace(trace, row=i, col=1)

        # Update layout properties
        fig.update_layout(height=400 * len(months), width=800, showlegend=True,
                          title_text=f'Tiempo entre inicio de corte Mes: {month}')

    return fig






def plot_time(df):
    from datetime import datetime, date, time
    import plotly.graph_objects as go
    import plotly.subplots as sp
    import pandas as pd
    # Convert 'Time' column to datetime.time
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

    # Create a dummy date and combine it with time to create a datetime object
    dummy_date = date.today()
    df['Datetime'] = df['Time'].apply(lambda t: datetime.combine(dummy_date, t))

    # Convert 'Date' column to a datetime data type
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the DataFrame index
    df.set_index('Date', inplace=True)

    # Retrieve the unique months in the DataFrame
    months = df.index.to_period('M').unique()

    fig = sp.make_subplots(rows=len(months), cols=1)

    for i, month in enumerate(months, start=1):
        # Create a subset of the DataFrame for the current month
        df_month = df[df.index.to_period('M') == month]

        # Calculate the mean time for this month
        mean_datetime = df_month['Datetime'].mean()

        # Create a line plot with markers for this month
        scatter = go.Scatter(x=df_month.index.day,
                             y=df_month['Datetime'],
                             mode='markers+lines',
                             name='hour',
                             )
        fig.add_trace(scatter, row=i, col=1)

        # Plot the mean time for this month
        mean_line = go.Scatter(x=[df_month.index.day.min(), df_month.index.day.max()],
                               y=[mean_datetime, mean_datetime],
                               mode='lines',
                               name=f'Mean: {mean_datetime.time()}',
                               line=dict(color='red'))

        fig.add_trace(mean_line, row=i, col=1)

        # Update xaxis and yaxis properties
        fig.update_xaxes(title_text='Day', row=i, col=1)
        fig.update_yaxes(title_text='Hour of day', tickformat='%H:%M:%S',
                         range=['00:00:00', '23:59:59'], row=i, col=1)

        fig.update_layout(height=400 * len(months), width=800, showlegend=True,
                          title_text=f'Tiempo de inicio del Laser for {month.strftime("%B %Y")}')

    return fig

import datetime


def plot_segments_multicolumn(df, columns):
    """
    Plots segments in a multicolumn layout from the given DataFrame.

    :param df: The DataFrame to be processed. It should have 'Message' and 'Timestamp' columns.
               'Timestamp' should already be in the datetime format.
    :param columns: Number of columns in the plot layout.
    :return: None
    """
    df['day_month'] = df['Timestamp'].dt.strftime('%m-%d')

    df['flag'] = df['Message'].str.contains('Open File').cumsum()

    # Calculating rows needed for the plots
    rows = math.ceil(len(df['flag'].unique()) / columns)

    fig, axs = plt.subplots(rows, columns, figsize=(15, 6 * rows))

    # Reshaping in case of one single row
    axs = axs.reshape(-1, columns)

    for i, segment in enumerate(df['flag'].unique()):
        segment_df = df[df['flag'] == segment].copy()
        file_name = segment_df[segment_df['Message'].str.contains('Open File')]['Message'].str.split("\\").str[-1]
        file_name = file_name.iloc[0] if not file_name.empty else ""
        segment_date = segment_df['day_month'].iloc[0]
        segment_df[
            'machining_seconds'] = segment_df['Message'].str.extract('Total machining: (\d+.?\d*) s').astype(float)
        mask = segment_df['Message'].str.contains("Layer: 1")
        segment_df.loc[mask, 'machining_seconds'] = 0
        ax = axs[i // columns, i % columns]
        ax.plot(segment_df['Timestamp'], segment_df['machining_seconds'])
        layer1_entries = segment_df[mask]['Timestamp']
        for date in layer1_entries:
            ax.axvline(x=date, color='red', linewidth=0.5)
        ax.set_title(f"Archivo:{file_name}, \nFecha: {segment_date}", fontsize=8)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Machining Time in seconds')
        ax.tick_params(axis='x', rotation=45)

    # If total subplots are not a multiple of columns then delete remaining axes
    for i in range(len(df['flag'].unique()), rows * columns):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()

    # save the figure
    now = datetime.datetime.now()
    plt.savefig(f'plots/Layer1_{now.year}_{now.month}.png')

    plt.show()




def create_barplot(df, x_col, y_col, x_title, y_title):
    # Sort dataframe
    df_sort = df.sort_values(y_col, ascending=True).reset_index(drop=True)
    df_sort['text_label'] = df_sort[y_col].round(2).astype(str)

    fig = px.bar(df_sort, x=x_col, y=y_col,
                 text='text_label',
                 color=x_col,
                 # pattern_shape=x_col,
                 # pattern_shape_sequence=[".", "x", "+"],
                 labels={x_col: x_title, y_col: y_title},
                 title=f'{x_title} by {y_title}')
    fig.update_layout(autosize=False, width=1200, height=800)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    # Customize aspect
    fig.update_layout(autosize=False)

    fig.update_xaxes(tickangle=45)

    # Save the plot
    # fig.write_image(f'plots/tiempo_programa_{"your_month"}.png')

    # Show the plot
    return fig