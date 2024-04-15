import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

REGIONS = ['Portugal', 'Guangdong', 'Macau']
input_cols = ['NewCases']
input_cols_all = [c + '_' + r for c in input_cols for r in REGIONS]
output_col = ['NewCases']
MOV_AVG_WIN = 3
time_lag = 1
DAY_MODE = '1D'

'''
Omicron first reported on 2021-11-24. https://www.who.int/news/item/26-11-2021-classification-of-omicron-(b.1.1.529)-sars-cov-2-variant-of-concern
We minus the "MOV_AVG_WIN" because these rows has not enough previous data to compute the moving average
'''
OMICRON_START_COLS = 672 - MOV_AVG_WIN


def load_datasets(region, use_all=False):
    # Load data from Excel
    data_df = pd.read_excel('data/input/c19_data_for_FL.xlsx', sheet_name=region, index_col='Date')
    if use_all:
        # Use data from all regions as input
        dfs = []
        # new_input_cols = []
        for r in REGIONS:
            df = pd.read_excel('data/input/c19_data_for_FL.xlsx', sheet_name=r, index_col='Date')
            # new_input_cols.extend([col + '_' + r for col in input_cols])

            rename_columns = {old: new for old, new in [(col, col + '_' + r) for col in input_cols]}
            if r == region:
                # Avoid rename target columns for the current region
                df_rename = df.rename(columns=rename_columns, inplace=False)
                df = pd.merge(df, df_rename, left_index=True, right_index=True)
            else:
                df = df[input_cols]
                df.rename(columns=rename_columns, inplace=True)
            dfs.append(df)
        data_df = pd.concat([df for df in dfs], axis=1)
        # input_cols = new_input_cols

    data_df = data_df[OMICRON_START_COLS:]  # Ignore data prior to the first reported Omicron case

    # Compute moving average
    for key in data_df.keys():
        data_df[key] = data_df[key].rolling(window=MOV_AVG_WIN).mean()
    data_df = data_df[MOV_AVG_WIN - 1:]  # Delete the data who has no enouth previous data to complete moving average

    # print(f'data_df={data_df}')
    # data_df = data_df[0:experiment_index * EXPERIMENT_STEP_SIZE + time_lag]

    data_df = data_df.resample(DAY_MODE).mean()  # Resample data on weekly basis, taking mean
    data = data_df[input_cols_all if use_all else input_cols].values
    target = data_df[output_col].values

    # Align the data
    data = data[0:-time_lag]
    target = target[time_lag:]
    data_df = data_df[time_lag:]

    print(f'load_datasets: data.shape={data.shape}')
    print(f'load_datasets: target.shape={target.shape}')

    # Normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    y = scaler.fit_transform(target.reshape(-1, 1)).flatten()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # Convert to PyTorch tensors. add unsqueeze(1) to match the expected input shape
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return data_df.index, X_train, X_test, y_train, y_test, scaler
