import numpy as np
import pandas as pd
import portalocker
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from config import CONFIG
from dataset import Dataset

input_cols = CONFIG['input_cols']
input_cols_all = [c + '_' + r for c in input_cols for r in CONFIG['regions']]
output_col = CONFIG['output_col']
MOV_AVG_WIN = CONFIG['moving_average']
DAY_MODE = '1D'

'''
Omicron first reported on 2021-11-24. https://www.who.int/news/item/26-11-2021-classification-of-omicron-(b.1.1.529)-sars-cov-2-variant-of-concern
We minus the "MOV_AVG_WIN" because these rows has not enough previous data to compute the moving average
'''
# OMICRON_START_COLS = 672 - MOV_AVG_WIN
OMICRON_START_COLS = 24 - MOV_AVG_WIN - CONFIG['input_len']


def load_datasets(region, use_all=False, ahead=0):
    # Load data from Excel
    regions = CONFIG['regions'] if use_all else [region]

    index = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    return_scaler = None
    local_y_train = []

    date_index = []

    for r in regions:
        with open(CONFIG['data_path'], 'r+b') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            data_df = pd.read_excel(f, sheet_name=r, index_col='Date', engine='openpyxl')
            portalocker.unlock(f)  # 解锁文件
        data_df = data_df[OMICRON_START_COLS:]  # Ignore data prior to the first reported Omicron case

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
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        target = scaler.fit_transform(target.reshape(-1, 1)).flatten()

        if r == region:
            return_scaler = scaler

        # Align the data
        # Use the previous few days sliding windowas input, e.g. 7 days, to predict the next day
        # TODO Use figure to show the performance difference
        features = []
        # 从第8天开始，每个时间步的特征包含过去7天的数据
        input_len = CONFIG['input_len']
        for i in range(input_len, len(data) - ahead):
            feature = data[i - input_len:i]  # 过去7天的数据  # Alternative: use np.clip
            features.append(feature.flatten())
        X = np.array(features)
        y = target[input_len + ahead:]
        data_df = data_df[input_len:]

        for i in range(ahead, len(data_df.index.values)):
            date_index.append(data_df.index.values[i])

        # print(f'load_datasets: data.shape={X.shape}')
        # print(f'load_datasets: target.shape={y.shape}')

        # Train-test split

        if r == region:
            split_index = int(len(X) * 0.8)
            X_train_temp = X[0:split_index]
            y_train_temp = y[0:split_index]
            X_test_temp = X[split_index:]
            y_test_temp = y[split_index:]

            local_y_train.extend(y_train_temp)

            X_train.extend(X_train_temp)
            y_train.extend(y_train_temp)

            X_test.extend(X_test_temp)
            y_test.extend(y_test_temp)
        else:
            X_train.extend(X)
            y_train.extend(y)

    # Shuffle the training data
    X_train_origin, y_train_origin = X_train, y_train  # Remember the origin data for drawing plots later
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Convert to PyTorch tensors. add unsqueeze(1) to match the expected input shape
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    X_train_origin = torch.tensor(X_train_origin, dtype=torch.float32).unsqueeze(1)
    y_train_origin = torch.tensor(y_train_origin, dtype=torch.float32)
    local_y_train = torch.tensor(local_y_train, dtype=torch.float32)

    if use_all:
        print(f'date_index={date_index}')
        index = [i for i in range(len(y_train) + len(y_test) + ahead)]
    else:
        index = data_df.index

    print(f'Dataset loaded')
    dataset = Dataset(index, X_train, X_test, y_train, y_test, return_scaler, X_train_origin, y_train_origin, date_index, local_y_train)
    return dataset
