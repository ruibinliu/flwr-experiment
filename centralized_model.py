import time
from datetime import datetime
from math import sqrt
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import CONFIG
from data_loader import load_datasets
from model import EarlyStopping, LSTM
from performance import save_performance, smape

MODEL_NAME = 'C-LSTM'
time_str = datetime.now().strftime('%m/%d_%H%M%S')
output_dir = f'data/output/{time_str}_centralized'
os.makedirs(output_dir)

# Set random seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)


def lstm(region, dataset, ahead=0):
    start_time = time.time()

    # Initialize model, loss function, and optimizer
    model = LSTM(CONFIG['input_len'], CONFIG['hidden_size'], CONFIG['num_layers'], len(CONFIG['output_col']),
                 dropout=CONFIG['dropout'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    model.train()
    train_losses = []
    early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=False)
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        outputs = model(dataset.x_train)
        optimizer.zero_grad()
        # add unsqueeze(1) to match the expected target shape
        loss = criterion(outputs, dataset.y_train.unsqueeze(1))
        loss.backward()
        # Apply model changes
        optimizer.step()
        train_losses.append(loss.item())

        # Check EarlyStopping
        early_stopping(loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(early_stopping.path))

    end_time = time.time()
    print(f'{MODEL_NAME} training time: {round(end_time - start_time, 3)} seconds.')

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        y_pred_train_origin = model(dataset.X_train_origin)
        y_pred_test = model(dataset.x_test)

    # Denormalize predictions
    y_train_origin = dataset.scaler.inverse_transform(dataset.y_train_origin.detach().numpy().reshape(-1, 1)).flatten()
    y_train = dataset.scaler.inverse_transform(dataset.y_train.detach().numpy().reshape(-1, 1)).flatten()
    y_test = dataset.scaler.inverse_transform(dataset.y_test.unsqueeze(1).numpy().reshape(-1, 1)).flatten()
    y_pred_train_origin = dataset.scaler.inverse_transform(y_pred_train_origin.detach().numpy().reshape(-1, 1)).flatten()
    y_pred_test = dataset.scaler.inverse_transform(y_pred_test.detach().numpy().reshape(-1, 1)).flatten()

    y_pred_train_origin = pd.Series(y_pred_train_origin, index=dataset.index[ahead:len(dataset.x_train) + ahead], name=f'{MODEL_NAME}_ahead{ahead + 1}')
    y_pred_test = pd.Series(y_pred_test, index=dataset.index[len(dataset.x_train) + ahead:], name=f'{MODEL_NAME}_ahead{ahead + 1}')
    df_pred = pd.concat([pd.DataFrame(y_pred_train_origin), pd.DataFrame(y_pred_test)])
    df_pred['date_index'] = dataset.date_index

    loss = criterion(torch.tensor(y_pred_test.to_numpy()), torch.tensor(y_test))
    rmse = sqrt(mean_squared_error(y_test, y_pred_test))
    rmse_normalized = rmse / max(max(y_train), max(y_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    mae_normalized = mae / max(max(y_train), max(y_test))
    s_mape = smape(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f'Test Loss: RMSE={rmse:.2f}, MAE={mae:.2f}, SMAPE={s_mape}:.2f')

    end_time = time.time()
    print(f'{MODEL_NAME} testing time: {round(end_time - start_time, 3)} seconds.')

    index_of_dataset_begin = ''
    index_of_dataset_end = ''
    index_of_train_begin = ''
    index_of_train_end = ''
    index_of_test_begin = ''
    index_of_test_end = ''

    save_performance(output_dir, region,
                     rmse, rmse_normalized,
                     mae, mae_normalized,
                     s_mape, r2, MODEL_NAME,
                     index_of_dataset_begin, index_of_dataset_end,
                     index_of_train_begin, index_of_train_end,
                     index_of_test_begin, index_of_test_end, ahead)

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(dataset.index[ahead:], np.concatenate((y_train_origin, y_test), axis=0), label='Reported cases', c='black')
    # plt.plot(data_df.index, torch.cat((y_train, y_test), dim=0).numpy(), label='Data')
    axs[0].plot(y_pred_train_origin, label=f'{MODEL_NAME} (Train)')
    axs[0].plot(y_pred_test, label=f'{MODEL_NAME} (Test)')
    axs[0].set_title(f'{MODEL_NAME} prediction for {region}')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Number of reported cases')
    axs[0].legend()

    axs[1].plot(train_losses, label='Loss')
    axs[1].set_title('Training loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAE')
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{region}-{MODEL_NAME}.png')
    plt.cla()
    plt.close()

    return df_pred


if __name__ == '__main__':
    for region in CONFIG['regions']:
        results = []
        for ahead in range(0, CONFIG['output_len']):
            t0 = time.time()
            dataset = load_datasets(region, use_all=True, ahead=ahead)
            df = lstm(region, dataset, ahead=ahead)
            results.append(df)
            t1 = time.time()
            print(f'{MODEL_NAME}({region}) cost {round(t1 - t0, 3)} seconds.')

        # 合并 DataFrame
        df = pd.concat(results, axis=1)
        output_file = f'{output_dir}/predit_data-{MODEL_NAME}.xlsx'
        mode = 'a' if os.path.exists(output_file) else 'w'
        with pd.ExcelWriter(output_file, engine='openpyxl', mode=mode) as writer:  # 使用writer避免覆盖已有的sheet
            df.to_excel(writer, sheet_name=region)
