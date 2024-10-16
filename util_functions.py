import codecs
import numpy as np
import re
import datetime


def rtf_to_dataframe(file_path):
    # Initialize an empty dataframe
    df = pd.DataFrame(columns=["Timestamp", "Message"])

    # Read RTF file
    with codecs.open(file_path, "r", "utf-8") as file:
        content = file.read()

        # Split the file content into lines
        rows = content.split("\\par")

        # Regex pattern for timestamp and message
        timestamp_pattern = r'\(([0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\)'
        message_pattern = r'\\cf[0-9](.*?)\\cf0'  # Escape the backslashes

        for row in rows:
            # Find timestamp and message
            timestamp = re.search(timestamp_pattern, row)
            message = re.search(message_pattern, row)

            # Append row to dataframe
            if timestamp and message:
                df.loc[len(df)] = [timestamp.group(1), message.group(1)]

    return df


def planchas_cortadas(df, message_filter, time_format='%m/%d %H:%M:%S', diff_minutes=3):
    ''' Filter messages that contain a certain string, reindex, and count monthly occurrences '''

    # Filter df and create a copy
    df_filtered = df[df['Message'].str.contains(message_filter, case=False, na=False)].copy()

    # Attach a year for datetime conversion
    df_filtered['Timestamp'] = pd.to_datetime('1900/' + df_filtered['Timestamp'])

    # Use Timestamp as Index
    df_filtered.set_index('Timestamp', inplace=True)

    # Compute the boolean mask for <= diff_minutes difference
    mask = (df_filtered.index.to_series().diff() <= pd.Timedelta(minutes=diff_minutes))

    # Invert the mask (~), so True means we keep the row
    df_filtered = df_filtered.loc[~mask]

    # Calculate number of messages per month
    messages_per_month = df_filtered.resample('ME').count()['Message']

    for index, value in messages_per_month.items():
        print(f"Month: {index.month}, Planchas Cortadas: {value}")

    return messages_per_month


def planchas_cortadas_day(df, message_filter, time_format='%m/%d/%Y %H:%M:%S', diff_minutes=3):
    ''' Filter messages that contain a certain string, reindex, and count daily occurrences '''
    import datetime

    current_year = datetime.datetime.now().year

    # Filter df and create a copy
    df_filtered = df[df['Message'].str.contains(message_filter, case=False, na=False)].copy()

    # Attach current year for datetime conversion
    df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'] + f'/{current_year}')

    # Use Timestamp as Index
    df_filtered.set_index('Timestamp', inplace=True)

    # Compute the boolean mask for <= diff_minutes difference
    mask = (df_filtered.index.to_series().diff() <= pd.Timedelta(minutes=diff_minutes))

    # Invert the mask (~), so True means we keep the row
    df_filtered = df_filtered.loc[~mask]

    # Calculate number of messages per day
    messages_per_day = df_filtered.resample('D').count()['Message']

    # Transform series into a DataFrame with 'Date' and 'Planchas Cortadas' columns
    messages_per_day = messages_per_day.reset_index()
    messages_per_day.columns = ['Date', 'Planchas Cortadas']

    return messages_per_day



#
# def read_all_rtf_in_dir(directory_path):
#     files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.rtf')]
#     dfs = []
#     for num, filename in enumerate(files):
#         pp ='{}/{} : {}'.format(num, len(files), filename)
#         print(pp)
#         full_path = os.path.join(directory_path, filename)
#         df = rtf_to_dataframe(full_path)
#         dfs.append(df)
#     # If you want to concatenate all dataframes into one:
#     result = pd.concat(dfs, ignore_index=True)
#     return result


import pandas as pd
import os


def read_all_rtf_in_dir(directory_path, save_folder, replace_flag):
    """
    This function reads all .rtf files from a given directory,
    converts each file to a DataFrame and then concatenates all
    DataFrames into a single DataFrame. The final DataFrame is then
    saved to a .csv file in a specific folder. If the .csv file already
    exists, the function can either replace it with a new DataFrame or
    load the existing DataFrame from the file, based on the 'replace_flag'.

    Args:
        directory_path (str): The path of the directory from which .rtf files are to be read.

        save_folder (str): The name of the directory where the DataFrame will be saved
        as a .csv file. If this directory does not exist, it will be created.

        replace_flag (bool): A flag to determine behaviour when the .csv file already exists.
        If True, any existing .csv file will be replaced with a new DataFrame. If False,
        the existing DataFrame will be loaded from the .csv file.

    Returns:
        result (DataFrame): The final DataFrame. Either loaded from the .csv file if it
        exists and replace_flag is False, or newly created from the .rtf files.
    """

    # Checks if save_folder exists, if not creates it
    save_folder_full_path = os.path.join(directory_path, save_folder)
    if not os.path.exists(save_folder_full_path):
        os.makedirs(save_folder_full_path)

    # Save file path
    save_file_path = os.path.join(save_folder_full_path, "saved_df.csv")

    # Checks if replace_flag is True or saved file doesn't exist
    if replace_flag or not os.path.isfile(save_file_path):
        files = [f for f in os.listdir(directory_path)
                 if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.rtf')]
        dfs = []
        for num, filename in enumerate(files):
            pp = '{}/{} : {}'.format(num, len(files), filename)
            print(pp)
            full_path = os.path.join(directory_path, filename)
            df = rtf_to_dataframe(full_path)
            dfs.append(df)
        # If you want to concatenate all dataframes into one:
        result = pd.concat(dfs, ignore_index=True)
        # Save df to a file
        result.to_csv(save_file_path)
    else:
        # Read df from a file
        result = pd.read_csv(save_file_path)

    return result



def filter_by_message_and_extract(df):
    # Create a copy of the original df
    df_copy = df.copy()

    # Filter the relevant rows
    filtered_df = df_copy[df_copy['Message'].str.contains('open file|Total machining', case=False, na=False)]

    # Extract the number from 'Total machining' message
    def extract_number(message):
        match = re.search(r'Total machining: (\d+\.\d+)', message)
        return float(match.group(1)) if match else np.nan

    # Initialize an empty DataFrame for results
    results = pd.DataFrame(columns=filtered_df.columns)

    # Initialize an empty DataFrame to temporarily store 'Total machining' rows
    temp_df = pd.DataFrame(columns=filtered_df.columns)

    # Flag to indicate if we are between 'Open File' messages
    between_open_files = False

    # Iterate over each row
    for i, row in filtered_df.iterrows():
        message = row['Message']
        if 'Open File' in message:
            if not temp_df.empty:
                # Find the row with the max 'Total machining' value in temp_df and add it to results
                max_row = temp_df.loc[temp_df['Total machining'].idxmax()]
                results = pd.concat([results, pd.DataFrame(max_row).T])
                # Clear temp_df
                temp_df = pd.DataFrame(columns=filtered_df.columns)
            between_open_files = True
            results = pd.concat([results, pd.DataFrame(row).T])

        elif between_open_files and 'Total machining' in message:
            # Extract the 'Total machining' value and add the row to temp_df
            row['Total machining'] = extract_number(message)
            temp_df = pd.concat([temp_df, pd.DataFrame(row).T])

    # Handle the case where 'Total machining' rows exist after the last 'Open File'
    if not temp_df.empty:
        max_row = temp_df.loc[temp_df['Total machining'].idxmax()]
        results = pd.concat([results, pd.DataFrame(max_row).T])

    # Reset the index
    results.reset_index(drop=True, inplace=True)

    return results


def filter_and_replace(df):
    df_copy = df.copy()

    # Create boolean masks for rows where 'Message' contains 'open file' or 'Total machining'
    mask_open_file = df_copy['Message'].str.contains('open file', case=False, na=False)
    mask_total_machining = df_copy['Message'].str.contains('Total machining', case=False, na=False)

    # Apply the mask_open_file and replace the 'Message' with the part after the last '\\'
    df_copy.loc[mask_open_file, 'Message'] = df_copy.loc[mask_open_file, 'Message'].str.split('\\\\').str[-1]

    # Filter the dataframe for 'open file' or 'Total machining'
    filtered_df = df_copy[mask_open_file | mask_total_machining]

    return filtered_df

def create_dataframe_from_filtered_data_and_sort(filtered_df):
    data = []  # Initialize an empty list to store the rows data

    prev_row = None  # Placeholder for the previous row

    for _, row in filtered_df.iterrows():
        if 'Total machining' in row['Message']:
            if prev_row is not None:  # Ensure that there is a previous row
                temp_dict = {'Hora': prev_row['Timestamp'], 'Programa': prev_row['Message'],
                             'Tiempo Maquinado (Seg)': row['Total machining']}
                data.append(temp_dict)
        prev_row = row  # Keep track of the previous row

    new_df = pd.DataFrame(data)  # Create a new DataFrame from the data list

    # Sort by 'Programa' and 'Tiempo Maquinado' and drop duplicates on 'Programa'
    final_df = (new_df.sort_values(['Programa', 'Tiempo Maquinado (Seg)'],
                                   ascending=[True, False])
                      .drop_duplicates('Programa', keep='first'))

    # Remove rows where 'Programa' is 'Cancel Open File'
    final_df = final_df[final_df['Programa'] != 'Cancel Open File']

    # Sort by 'Hora'
    final_df = final_df.sort_values('Hora')

    return final_df



def compute_total_time(df):
    tiempo_total = df[df['Message'].str.contains('Total machining', case=False, na=False)].copy()

    # Removing text from Messages and extract time values
    tiempo_total['Message'] = tiempo_total['Message'].str.replace('Total machining:', '')
    tiempo_total['Time'] = pd.to_numeric(tiempo_total['Message'].str.extract('(\d+.\d+) ')[0])

    tiempo_total['Time_Hours'] = tiempo_total['Time'] / 3600

    # Attach a dummy year for date-time conversion
    tiempo_total['Timestamp'] = pd.to_datetime('1900/' + tiempo_total['Timestamp'])

    tiempo_total.set_index('Timestamp', inplace=True)

    # Grouping by month and calculating the total time for each month
    # 'ME' stands for month end frequency
    total_time_by_month = tiempo_total.resample('ME').sum()['Time_Hours']

    for index, value in total_time_by_month.items():
        pp = 'Month: {}, Total Hours: {}'.format(index.month, round(value, 2))
        print(pp)

    return total_time_by_month


def group_by_date(df):
    # Filter the dataframe
    tiempo_total = df[df['Message'].str.contains('Total machining', case=False, na=False)].copy()  # Explicit copy here

    # Removing text from Messages and extract time values
    tiempo_total['Message'] = tiempo_total['Message'].str.replace('Total machining:', '')
    tiempo_total['Time'] = pd.to_numeric(tiempo_total['Message'].str.extract(r'(\d+.\d+) ')[0])

    # Extract the date part from 'Timestamp' without adding "2024-"
    tiempo_total['Timestamp'] = pd.to_datetime(tiempo_total['Timestamp'])
    tiempo_total['Date'] = tiempo_total['Timestamp'].dt.date

    # Group by 'Date' and calculate the sum of 'Time' for each date
    grouped_tiempo_total = tiempo_total.groupby('Date')['Time'].sum().reset_index()

    # Divide each row of 'Time' column by 120
    grouped_tiempo_total['Time'] = grouped_tiempo_total['Time'].apply(lambda x: x / 60)

    return grouped_tiempo_total



def time_between_placas(df, filters):
    # Create a boolean Series for filtering
    mask = pd.Series(False, index=df.index)
    for filter in filters:
        mask |= df['Message'].str.contains(filter)

    df_filtered = df[mask]

    final_df = pd.DataFrame()

    # Iterate over the DataFrame
    for i in range(len(df_filtered) - 1):
        # Check if a row contains the first filter term and the next row contains the second filter term
        if filters[0] in df_filtered.iloc[i]['Message'] and filters[1] in df_filtered.iloc[i + 1]['Message']:
            # Append both rows to final_df using pandas concat
            final_df = pd.concat([final_df, df_filtered.iloc[[i, i + 1]]])

    final_df.reset_index(drop=True, inplace=True)

    # Sort the DataFrame by 'Timestamp'
    final_df = final_df.sort_values('Timestamp')

    # Convert 'Timestamp' to datetime format
    final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Create a 'Date' column which is the date part of the 'Timestamp'
    final_df['Date'] = final_df['Timestamp'].dt.date

    # Group by the 'Date' column and calculate the difference within each group
    final_df['Timestamp_Diff'] = final_df.groupby('Date')['Timestamp'].diff()

    # Reset the index for the Timestamp_Diff series and drop na
    timestamp_diff_df = final_df[['Date', 'Timestamp_Diff']].dropna().reset_index(drop=True)

    # Drop the rows where 'Timestamp_Diff' is less than 2 minutes
    timestamp_diff_df = timestamp_diff_df[timestamp_diff_df['Timestamp_Diff'] > pd.Timedelta(minutes=2)]

    # Convert 'Timestamp_Diff' from timedelta object to int, showing minutes
    timestamp_diff_df['Timestamp_Diff'] = (timestamp_diff_df['Timestamp_Diff'].dt.total_seconds() / 60).astype(int)

    # Drop the rows where 'Timestamp_Diff' is greater than 600
    timestamp_diff_df = timestamp_diff_df[timestamp_diff_df['Timestamp_Diff'] <= 600]

    return final_df, timestamp_diff_df

def first_occurrence_per_date(df, column, search_str):
    from datetime import datetime
    # Filter rows which contain search string
    df_search = df[df[column].str.contains(search_str, case=False, na=False)].copy()

    # Convert 'Timestamp' to datetime
    df_search['Timestamp'] = pd.to_datetime(df_search['Timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Assign current year
    now = datetime.now()
    df_search.loc[:, 'Timestamp'] = df_search['Timestamp'].map(lambda dt: dt.replace(year=now.year))

    # Extract date and time
    df_search['Date'] = [d.date() for d in df_search['Timestamp']]
    df_search['Time'] = [d.time() for d in df_search['Timestamp']]

    # Group by date, drop 'Timestamp' and get the first occurrence per date
    df_first_occurrence = df_search.sort_values('Time').groupby('Date', as_index=False).first()
    df_first_occurrence = df_first_occurrence.drop(columns=['Timestamp'])

    return df_first_occurrence
from datetime import datetime
def get_months_and_years_since(date_str):

    initial_date = datetime.strptime(date_str, "%d/%m/%Y")
    current_date = datetime.now()

    months = set()
    years = set()

    while initial_date <= current_date:
        months.add(initial_date.month)
        years.add(initial_date.year)
        initial_date = add_months(initial_date, 1)

    # Separate current month and year
    cur_month = current_date.month
    cur_year = current_date.year

    return sorted(list(months)), sorted(list(years)), cur_month, cur_year

def add_months(date, months):
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, [31,29,31,30,31,30,31,31,30,31,30,31][month-1])
    return datetime(year, month, day)
def extract_month_year(df):
    df['Date'] = pd.to_datetime(df['Date'])

    # Create month and year columns
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    return df


def get_surrounding_rows(df, messages, column, x):
    """
    Function to return x rows before and after each row
    containing any of the 'messages' in the specified 'column'
    """
    # Concatenate all messages with '|' to use as OR in regular expressions
    messages_regex = '|'.join(messages)

    matches = df[df[column].str.contains(messages_regex)]

    surrounding_rows = pd.DataFrame()

    # retrieve x rows before and after for each matching row
    for index in matches.index:
        start = max(0, index - x)
        end = index + x + 1  # +1 because DataFrame slicing is upper-bound exclusive
        surrounding_rows = pd.concat([surrounding_rows, df.iloc[start:end]])
    surrounding_rows = surrounding_rows.sort_values(by='Timestamp')
    return surrounding_rows.drop_duplicates()

def filter_open_file_messages(df, month):
    """
    Filters a DataFrame based on 'Open File' messages and a certain month.

    :param df: DataFrame to filter. It should include 'Message' and 'Timestamp' columns.
               The 'Timestamp' should be in string format representing datetime.
    :param month: The target month as an integer. For example, 1 for January, 2 for February, etc.
    :return: The filtered DataFrame.
    """
    df['Message'] = df['Message'].astype(str)
    file_messages = df['Message'].str.contains('Open File')
    shifted_file_messages = df['Message'].shift(-1).fillna("NotAnOpenFile")
    message_filter = (~file_messages) | (file_messages & ~shifted_file_messages.str.contains('Open File'))
    filtered_df = df.loc[message_filter]

    # Convert 'Timestamp' to datetime object
    filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])

    # Filter rows based on the provided month
    filtered_df = filtered_df[filtered_df['Timestamp'].dt.month == month]

    return filtered_df

def drop_open_file_duplicates(df):
    """
    Processes a DataFrame and drops duplicates of 'Open File' messages,
    keeping only the first occurrence.

    :param df: The DataFrame to be processed. It should have a 'Message' column.
    :return: The processed DataFrame without duplicate 'Open File' messages.
    """
    # Create a mask where True indicates rows with 'Open File' in 'Message'
    open_file_mask = df['Message'].str.contains('Open File')

    # Assign a unique id to each 'Open File' message
    df.loc[open_file_mask, 'open_file_id'] = df.loc[open_file_mask, 'Message'].rank(method='dense')

    # Keep only the first occurrence of each 'Open File' message
    df = df.loc[~df.duplicated(subset='open_file_id') | df['open_file_id'].isna()].copy()

    # Drop the 'open_file_id' column
    df = df.drop(columns='open_file_id')

    return df

def total_machining_per_program(df):

    """
    Returns a DataFrame where each row represents one program and its total machining time.
    Retains the Timestamp of the 'Open File' message.

    :param df: DataFrame to process. It should include 'Message' and 'Timestamp' columns.
    :return: The processed DataFrame with columns 'timestamp', 'program', and 'total_machining'.
    """

    df = df.copy()

    mask = df['Message'].str.contains('Open File')
    programs_df = df.loc[mask, ['Timestamp', 'Message']].copy()
    programs_df['Message'] = programs_df['Message'].apply(lambda x: x.split('\\')[-1])

    df['machining'] = np.where(df['Message'].str.contains('Total machining'),
                               df['Message'].str.extract(r'Total machining: (\d+.\d+) s', expand=False), np.nan)
    df['machining'] = pd.to_numeric(df['machining'], errors='coerce')

    df['program'] = np.where(mask, df['Message'], np.nan)
    df['program'] = df['program'].apply(lambda x: x.split('\\')[-1] if isinstance(x, str) else x)
    df['program'] = df['program'].ffill()

    total_machining = df.groupby('program')['machining'].sum().reset_index()

    total_machining_df = programs_df.merge(total_machining, left_on='Message', right_on='program')
    total_machining_df.drop(columns=['Message'], inplace=True)
    total_machining_df.columns = ['timestamp', 'program', 'total_machining']
    total_machining_df['total_machining'] = round(total_machining_df['total_machining']/60, 2)

    # Assuming dataframe is your DataFrame and 'timestamp' is your timestamp column
    total_machining_df['timestamp'] = pd.to_datetime(
        total_machining_df['timestamp'])  # making sure it's in datetime format
    total_machining_df['timestamp'] = total_machining_df['timestamp'].dt.strftime(
        '%Y-%m-%d %H:%M:%S')  # formatting to your required format

    return total_machining_df


# def group_and_sum(df, group_column, sum_column):
#     output_df = df.groupby(group_column)[sum_column].sum().reset_index()
#     return output_df


def group_and_sum(df, timestamp_column, group_column, sum_column):
    # Creating a 'Date' column with year-month format:
    df['Date'] = pd.to_datetime(df[timestamp_column]).dt.to_period('M').dt.to_timestamp()

    output_df = df.groupby([group_column, 'Date'])[sum_column].sum().reset_index()

    return output_df


# def transform_data(df):
#     df['Time (min)'] = df['total_machining'] / df['Programas cortados']
#
#     df.drop(['timestamp', 'program', 'Horas Teoricas', 'Horas Reales',
#              'Diferencia', 'Programas cortados', 'total_machining'], axis=1, inplace=True)
#
#     df_grouped = df.groupby('Espesor').agg({'Longitude Corte (m)': 'sum', 'Time (min)': 'sum'})
#     df_reset = df_grouped.reset_index()
#
#     df_reset['Velocidad (m/min)'] = round(df_reset['Longitude Corte (m)'] / df_reset['Time (min)'], 2)
#     df_reset = df_reset.sort_values(by='Espesor', ascending=True)
#
#     return df_reset

import pandas as pd

import pandas as pd


def transform_data(df, timestamp_column):
    df['Time (min)'] = df['total_machining'] / df['Programas cortados']
    df['Date'] = pd.to_datetime(df[timestamp_column]).dt.to_period('M').dt.to_timestamp()

    df.drop(['timestamp', 'program', 'Programas cortados', 'total_machining'], axis=1, inplace=True)

    df_grouped = df.groupby(['Espesor', 'Date']).agg({'Longitude Corte (m)': 'sum', 'Time (min)': 'sum'})
    df_reset = df_grouped.reset_index()

    df_reset['Velocidad (m/min)'] = round(df_reset['Longitude Corte (m)'] / df_reset['Time (min)'], 2)
    df_reset = df_reset.sort_values(by='Espesor', ascending=True)

    return df_reset


def weighted_average_espesor(df):
    """
    Function to calculate the weighted average of 'Espesor' with 'Programas cortados' as weights
    :param df: DataFrame
    :return: float - weighted average of 'Espesor'
    """
    try:
        total_programs = df['Programas cortados'].sum()
        if total_programs == 0:
            return 0
        weighted_sum = (df['Espesor'] * df['Programas cortados']).sum()
        weighted_average = weighted_sum / total_programs
        return weighted_average
    except Exception as e:
        return 0



def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df