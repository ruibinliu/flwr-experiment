import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import CONFIG

input_cols = CONFIG['input_cols']
input_cols_all = [c + '_' + r for c in input_cols for r in CONFIG['regions']]
output_col = CONFIG['output_col']
MOV_AVG_WIN = 7
DAY_MODE = '1D'

'''
Omicron first reported on 2021-11-24. https://www.who.int/news/item/26-11-2021-classification-of-omicron-(b.1.1.529)-sars-cov-2-variant-of-concern
We minus the "MOV_AVG_WIN" because these rows has not enough previous data to compute the moving average
'''
OMICRON_START_COLS = 672 - MOV_AVG_WIN


def load_datasets(region, use_all=False):
    # Load data from Excel
    regions = CONFIG['regions'] if use_all else [region]

    index = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for r in regions:
        data_df = pd.read_excel('data/input/c19_data_for_FL.xlsx', sheet_name=r, index_col='Date')
        data_df = data_df[OMICRON_START_COLS:]  # Ignore data prior to the first reported Omicron case

#     # TODO merge all the X_train from all regions
#     # Use data from all regions as input
#     dfs = []
#     # new_input_cols = []
#     for r in CONFIG['regions']:
#         df = pd.read_excel('data/input/c19_data_for_FL.xlsx', sheet_name=r, index_col='Date')
#
#         df = df[OMICRON_START_COLS:]  # Ignore data prior to the first reported Omicron case
#         if r == region:
#
#         '''
#         rename_columns = {old: new for old, new in [(col, col + '_' + r) for col in input_cols]}
#         if r == region:
#             # Avoid rename target columns for the current region
#             df_rename = df.rename(columns=rename_columns, inplace=False)
#             df = pd.merge(df, df_rename, left_index=True, right_index=True)
#         else:
#             df = df[input_cols]
#             df.rename(columns=rename_columns, inplace=True)
#         '''
#         dfs.append(df)
#     data_df = pd.concat([df for df in dfs])
#     # input_cols = new_input_cols
# else:
#     data_df = pd.read_excel('data/input/c19_data_for_FL.xlsx', sheet_name=region, index_col='Date')
#     data_df = data_df[OMICRON_START_COLS:]  # Ignore data prior to the first reported Omicron case

    # Compute moving average
        for key in data_df.keys():
            data_df[key] = data_df[key].rolling(window=MOV_AVG_WIN).mean()
        data_df = data_df[MOV_AVG_WIN - 1:]  # Delete the data who has no enouth previous data to complete moving average

        # print(f'data_df={data_df}')
        # data_df = data_df[0:experiment_index * EXPERIMENT_STEP_SIZE + time_lag]

        data_df = data_df.resample(DAY_MODE).mean()  # Resample data on weekly basis, taking mean
        data = data_df[input_cols].values
        target = data_df[output_col].values

        # Normalize data
        # TODO check the min-max scaler, e.g. consider [0, 1000]
        scaler = MinMaxScaler()  # feature_range=(0, 1000)
        data = scaler.fit_transform(data)
        target = scaler.fit_transform(target.reshape(-1, 1)).flatten()

        # Align the data
        # Use the previous few days sliding window, e.g. 7 days, to predict the next day
        # TODO Use figure to show the performance difference
        features = []
        # 从第8天开始，每个时间步的特征包含过去7天的数据
        window_size = CONFIG['window_size']
        for i in range(window_size, len(data)):
            feature = data[i - window_size:i]  # 过去7天的数据  # Alternative: use np.clip
            features.append(feature.flatten())
        X = np.array(features)
        y = target[window_size:]
        data_df = data_df[window_size:]

        print(f'load_datasets: data.shape={X.shape}')
        print(f'load_datasets: target.shape={y.shape}')

        if r == region:
            # Train-test split
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            X_train.extend(X_train_temp)
            y_train.extend(y_train_temp)
            X_test.extend(X_test_temp)
            y_test.extend(y_test_temp)
        else:
            X_train.extend(X)
            y_train.extend(y)

    # Convert to PyTorch tensors. add unsqueeze(1) to match the expected input shape
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    if use_all:
        index = [i for i in range(len(y_train) + len(y_test))]
    else:
        index = data_df.index

    return index, X_train, X_test, y_train, y_test, scaler
