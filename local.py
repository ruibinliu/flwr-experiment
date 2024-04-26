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

MODEL_NAME = 'LSTM'
time_str = datetime.now().strftime('%m/%d_%H%M%S')
output_dir = f'data/output/{time_str}_local'
os.makedirs(output_dir)

# Set random seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)


def lstm(region, index, x_train, x_test, y_train, y_test, scaler):
    start_time = time.time()

    # Initialize model, loss function, and optimizer
    model = LSTM(CONFIG['window_size'], CONFIG['hidden_size'], CONFIG['num_layers'], len(CONFIG['output_col']),
                 dropout=CONFIG['dropout'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    model.train()
    train_losses = []
    early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=True)
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        outputs = model(x_train)
        optimizer.zero_grad()
        # add unsqueeze(1) to match the expected target shape
        loss = criterion(outputs, y_train.unsqueeze(1))
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
        y_pred_train = model(x_train)
        y_pred_test = model(x_test)

    # Denormalize predictions
    scaler = scaler
    y_train = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.unsqueeze(1).numpy().reshape(-1, 1)).flatten()
    y_pred_train = scaler.inverse_transform(y_pred_train.detach().numpy().reshape(-1, 1)).flatten()
    y_pred_test = scaler.inverse_transform(y_pred_test.detach().numpy().reshape(-1, 1)).flatten()

    df_pred = pd.concat(
        [pd.DataFrame(y_pred_train, index=index[:len(x_train)]), pd.DataFrame(y_pred_test, index=index[len(x_train):])])
    df_pred.to_excel(f'{output_dir}/predit_data-{region}-{MODEL_NAME}.xlsx')

    loss = criterion(torch.tensor(y_pred_test), torch.tensor(y_test))
    rmse = sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    s_mape = smape(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f'Test Loss: RMSE={rmse:.2f}, MAE={mae:.2f}, SMAPE={s_mape}:.2f')

    end_time = time.time()
    print(f'{MODEL_NAME} testing time: {round(end_time - start_time, 3)} seconds.')

    index_of_dataset_begin = index[0].strftime('%Y-%m-%d')
    index_of_dataset_end = index[- 1].strftime('%Y-%m-%d')
    index_of_train_begin = index_of_dataset_begin
    index_of_train_end = index[len(x_train) - 1].strftime('%Y-%m-%d')
    index_of_test_begin = index[len(x_train)].strftime('%Y-%m-%d')
    index_of_test_end = index_of_dataset_end

    save_performance(output_dir, region, rmse, mae, s_mape, r2, MODEL_NAME,
                     index_of_dataset_begin, index_of_dataset_end,
                     index_of_train_begin, index_of_train_end,
                     index_of_test_begin, index_of_test_end)

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(index, np.concatenate((y_train, y_test), axis=0), label='Reported cases', c='black')
    # plt.plot(data_df.index, torch.cat((y_train, y_test), dim=0).numpy(), label='Data')
    axs[0].plot(index[:len(y_pred_train)], y_pred_train, label='LSTM predicted (Train)')
    axs[0].plot(index[len(y_pred_train):], y_pred_test, label='LSTM predicted (Test)')
    axs[0].set_title('LSTM Prediction')
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
    plt.savefig(f'{output_dir}/{region}-lstm.png')
    plt.cla()
    plt.close()

    return model


if __name__ == '__main__':
    for region in CONFIG['regions']:
        index, x_train, x_test, y_train, y_test, scaler = load_datasets(region)
        lstm_model = lstm(region, index, x_train, x_test, y_train, y_test, scaler)
