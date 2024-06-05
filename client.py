import argparse
from datetime import datetime
from math import sqrt
import time
import os

from collections import OrderedDict
from typing import List, Tuple
import flwr as fl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn

from config import CONFIG
from model import LSTM, EarlyStopping
from performance import save_performance, smape
from data_loader import load_datasets

# Configure PyTorch
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

ahead = 1

MODEL_NAME = 'F-LSTM'


def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, region):
        self.region = region
        self.dataset = load_datasets(region, use_all=False, ahead=ahead)

        # Initialize model, loss function, and optimizer
        self.model = LSTM(CONFIG['input_len'], CONFIG['hidden_size'],
                          CONFIG['num_layers'], len(CONFIG['output_col']),
                          dropout=CONFIG['dropout'])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG['learning_rate'])

    def get_parameters(self, config):
        print(f'FlowerClient: get_parameters')
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print(f'FlowerClient: fit(config={config})')
        start_time = time.time()
        output_folder = config['output_dir']
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        set_parameters(self.model, parameters)

        # Set the model to training mode
        self.model.train()
        train_losses = []
        early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=False, path=f'checkpoint_{self.region}.pt')
        # Training loop
        for epoch in range(CONFIG['num_epochs']):
            outputs = self.model(self.dataset.x_train)
            self.optimizer.zero_grad()
            # add unsqueeze(1) to match the expected target shape
            loss = self.criterion(outputs, self.dataset.y_train.unsqueeze(1))
            loss.backward()
            # Apply model changes
            self.optimizer.step()
            train_losses.append(loss.item())

            # Check EarlyStopping
            early_stopping(loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(early_stopping.path))

        end_time = time.time()
        total_time = round(end_time - start_time, 1)
        print(f'Client fitting time is {total_time} seconds.')

        # Plot results
        plt.figure(figsize=(8, 8), dpi=150)
        print(f'FL-LSTM fit: len(self.train_losses)={len(train_losses)}')
        plt.plot(train_losses, label='Loss')
        plt.title(f'Training loss ({self.region})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        # Adjust layout
        # round = len(performance_result[region]) + 1

        plt.savefig(f'{output_folder}/{self.region}-{MODEL_NAME}-loss')
        plt.cla()
        plt.close()

        return get_parameters(self.model), len(self.dataset.x_train), {}

    def evaluate(self, parameters, config):
        # round = len(performance_result[region]) + 1

        print(f'FlowerClient: evaluate (region={self.region}): config={config}')
        output_folder = config['output_dir']
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        set_parameters(self.model, parameters)

        # loss, accuracy = test(model, testloader)
        # print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

        # loss, accuracy = test(self.model, self.valloader)
        dataset = self.dataset

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            y_pred_train_origin = self.model(dataset.X_train_origin)
            y_pred_test = self.model(dataset.x_test)

        # De-normalize predictions
        y_train = dataset.scaler.inverse_transform(dataset.y_train.detach().numpy().reshape(-1, 1)).flatten()
        y_train_origin = dataset.scaler.inverse_transform(dataset.y_train_origin.detach().numpy().reshape(-1, 1)).flatten()
        y_test = dataset.scaler.inverse_transform(dataset.y_test.unsqueeze(1).numpy().reshape(-1, 1)).flatten()
        y_pred_train_origin = dataset.scaler.inverse_transform(y_pred_train_origin.detach().numpy().reshape(-1, 1)).flatten()
        y_pred_test = dataset.scaler.inverse_transform(y_pred_test.detach().numpy().reshape(-1, 1)).flatten()

        y_pred_train_origin = pd.Series(y_pred_train_origin, index=dataset.index[ahead:len(dataset.x_train) + ahead], name=f'{MODEL_NAME}_ahead{ahead + 1}')
        y_pred_test = pd.Series(y_pred_test, index=dataset.index[len(dataset.x_train) + ahead:], name=f'{MODEL_NAME}_ahead{ahead + 1}')
        df_pred = pd.concat([pd.DataFrame(y_pred_train_origin),
                             pd.DataFrame(y_pred_test)])
        df_pred.to_excel(f'{output_folder}/predit_data-{self.region}-{MODEL_NAME}.xlsx')

        loss = self.criterion(torch.tensor(y_pred_test), torch.tensor(y_test))
        s_mape = smape(y_test, y_pred_test).astype(float)
        rmse = sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_normalized = rmse / max(max(y_train), max(y_test))
        mae = mean_absolute_error(y_test, y_pred_test).astype(float)
        mae_normalized = mae / max(max(y_train), max(y_test))
        r2 = r2_score(y_test, y_pred_test)
        print(f'Test Loss: RMSE={rmse:.2f}, MAE={mae:.2f}, SMAPE={s_mape:.2f}')

        # self.performance.append((rmse, mae, s_mape))
        # new_rows = pd.DataFrame({'RMSE': [rmse], 'MAE': [mae], 'SMAPE': [s_mape]})
        # performance_result[region] = pd.concat([performance_result[region], new_rows], ignore_index=True)

        # index_of_dataset_begin = self.index[0].strftime('%Y-%m-%d')
        # index_of_dataset_end = self.index[-1].strftime('%Y-%m-%d')
        # index_of_train_begin = index_of_dataset_begin
        # index_of_train_end = self.index[len(self.x_train) - 1].strftime('%Y-%m-%d')
        # index_of_test_begin = self.index[len(self.x_train)].strftime('%Y-%m-%d')
        # index_of_test_end = index_of_dataset_end
        index_of_dataset_begin = ''
        index_of_dataset_end = ''
        index_of_train_begin = ''
        index_of_train_end = ''
        index_of_test_begin = ''
        index_of_test_end = ''

        save_performance(output_folder, self.region,
                         rmse, rmse_normalized,
                         mae, mae_normalized,
                         s_mape, r2, MODEL_NAME,
                         index_of_dataset_begin, index_of_dataset_end,
                         index_of_train_begin, index_of_train_end,
                         index_of_test_begin, index_of_test_end,
                         ahead, config['server_round'])

        # Plot results
        plt.figure(figsize=(8, 8), dpi=150)
        plt.plot(self.dataset.index[ahead:], np.concatenate((y_train_origin, y_test), axis=0), label='Reported cases', c='black')
        plt.plot(y_pred_train_origin, label=f'{MODEL_NAME} (Train)')
        plt.plot(y_pred_test, label=f'{MODEL_NAME} (Test)')
        plt.title(f'{MODEL_NAME} prediction for {self.region}')
        plt.xlabel('Time')
        plt.ylabel('Number of reported cases')
        plt.legend()
        plt.savefig(f'{output_folder}/{self.region}-{MODEL_NAME}')
        plt.cla()
        plt.close()

        return float(loss), len(self.dataset.x_test), {"rmse": rmse, 'mae': mae, 'smape': s_mape}


REGIONS = {1: 'Portugal', 2: 'Guangdong', 3: 'Macau'}


def start_client(node_id, epochs, ahead_arg):
    f"""
    Start federated learning client.

    :param node_id: See ${REGIONS}
    :param epochs: LSTM epochs
    :param ahead_arg: How many days ahead is going to be predicted.
    :return:
    """
    global ahead
    ahead = ahead_arg

    region = REGIONS[node_id]
    CONFIG['num_epochs'] = epochs

    print(f'Flower client started for region [{region}]. epochs={epochs}, ahead={ahead}.')

    # Start Flower client
    fl.client.start_client(server_address="127.0.0.1:18080", client=FlowerClient(region).to_client())
    # fl.client.start_client(server_address="pearl.mlkd.tp.vps.inesc-id.pt:8080", client=FlowerClient().to_client())


if __name__ == "__main__":
    N_CLIENTS = 3

    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "-n",
        "--node-id",
        type=int,
        choices=range(1, N_CLIENTS + 1),
        required=True,
        help="Specify the Client ID",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        choices=range(1, 10000),
        required=True,
        help="Specify the max number of epoches",
    )
    parser.add_argument(
        "-a",
        "--ahead",
        type=int,
        choices=range(0, 7),
        required=True,
        help="Specify the number of days ahead",
    )
    args = parser.parse_args()
    CONFIG['num_epochs'] = args.epoch

    start_client(args.node_id, args.epoch, args.ahead)
