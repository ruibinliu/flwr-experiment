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
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# Create output folder
time_str = datetime.now().strftime('%m/%d_%H%M%S')
output_dir = f'data/output/{time_str}'
os.makedirs(output_dir)


MODEL_LSTM = 'LSTM'
MODEL_A_LSTM = 'A_LSTM'
MODEL_FL_LSTM = 'FL_LSTM'


def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, region):
        self.region = region
        self.index, self.x_train, self.x_test, self.y_train, self.y_test, self.scaler = load_datasets(region)

        # Initialize model, loss function, and optimizer
        self.model = LSTM(CONFIG['window_size'], CONFIG['hidden_size'], CONFIG['num_layers'], len(CONFIG['output_col']), dropout=CONFIG['dropout'])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG['learning_rate'])

    def get_parameters(self, config):
        print(f'FlowerClient: get_parameters')
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print(f'FlowerClient: fit')
        start_time = time.time()

        set_parameters(self.model, parameters)

        # Set the model to training mode
        self.model.train()
        train_losses = []
        early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=True, path=f'checkpoint_{self.region}.pt')
        # Training loop
        for epoch in range(CONFIG['num_epochs']):
            outputs = self.model(self.x_train)
            self.optimizer.zero_grad()
            # add unsqueeze(1) to match the expected target shape
            loss = self.criterion(outputs, self.y_train.unsqueeze(1))
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

        plt.savefig(f'{output_dir}/{self.region}-{MODEL_FL_LSTM}-loss')
        plt.cla()
        plt.close()

        return get_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        # round = len(performance_result[region]) + 1

        print(f'FlowerClient: evaluate (region={self.region}): config={config}')

        set_parameters(self.model, parameters)

        # loss, accuracy = test(model, testloader)
        # print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

        # loss, accuracy = test(self.model, self.valloader)

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            y_pred_train = self.model(self.x_train)
            y_pred_test = self.model(self.x_test)

        # De-normalize predictions
        y_train = self.scaler.inverse_transform(self.y_train.detach().numpy().reshape(-1, 1)).flatten()
        y_test = self.scaler.inverse_transform(self.y_test.unsqueeze(1).numpy().reshape(-1, 1)).flatten()
        y_pred_train = self.scaler.inverse_transform(y_pred_train.detach().numpy().reshape(-1, 1)).flatten()
        y_pred_test = self.scaler.inverse_transform(y_pred_test.detach().numpy().reshape(-1, 1)).flatten()

        df_pred = pd.concat([pd.DataFrame(y_pred_train, index=self.index[:len(self.x_train)]),
                             pd.DataFrame(y_pred_test, index=self.index[len(self.x_train):])])
        df_pred.to_excel(f'{output_dir}/predit_data-{self.region}-{MODEL_FL_LSTM}.xlsx')

        loss = self.criterion(torch.tensor(y_pred_test), torch.tensor(y_test))
        s_mape = smape(y_test, y_pred_test).astype(float)
        rmse = sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test).astype(float)
        print(f'Test Loss: RMSE={rmse:.2f}, MAE={mae:.2f}, SMAPE={s_mape:.2f}')

        # self.performance.append((rmse, mae, s_mape))
        # new_rows = pd.DataFrame({'RMSE': [rmse], 'MAE': [mae], 'SMAPE': [s_mape]})
        # performance_result[region] = pd.concat([performance_result[region], new_rows], ignore_index=True)

        index_of_dataset_begin = self.index[0].strftime('%Y-%m-%d')
        index_of_dataset_end = self.index[-1].strftime('%Y-%m-%d')
        index_of_train_begin = index_of_dataset_begin
        index_of_train_end = self.index[len(self.x_train) - 1].strftime('%Y-%m-%d')
        index_of_test_begin = self.index[len(self.x_train)].strftime('%Y-%m-%d')
        index_of_test_end = index_of_dataset_end

        save_performance(output_dir, self.region, rmse, mae, s_mape, MODEL_FL_LSTM,
                         index_of_dataset_begin, index_of_dataset_end,
                         index_of_train_begin, index_of_train_end,
                         index_of_test_begin, index_of_test_end)

        # Plot results
        plt.figure(figsize=(8, 8), dpi=150)
        plt.plot(self.index, np.concatenate((y_train, y_test), axis=0), label='Reported cases', c='black')
        plt.plot(self.index[:len(y_pred_train)], y_pred_train, label='FL-LSTM prediction (Train)')
        plt.plot(self.index[len(y_pred_train):], y_pred_test, label='FL-LSTM prediction (Test)')
        plt.title('FL-LSTM Prediction')
        plt.xlabel('Time')
        plt.ylabel('Number of reported cases')
        plt.legend()
        plt.savefig(f'{output_dir}/{self.region}-{MODEL_FL_LSTM}')
        plt.cla()
        plt.close()

        return float(loss), len(self.x_test), {"rmse": rmse, 'mae': mae, 'smape': s_mape}


REGIONS = {1: 'Portugal', 2: 'Guangdong', 3: 'Macau'}

if __name__ == "__main__":
    N_CLIENTS = 3

    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "-n",
        "--node-id",
        type=int,
        choices=range(1, N_CLIENTS + 1),
        required=True,
        help="Specifies the Client ID",
    )
    args = parser.parse_args()
    region = REGIONS[args.node_id]
    print(f'Flower client started for region [{region}].')

    # Start Flower client
    fl.client.start_client(server_address="127.0.0.1:18080", client=FlowerClient(region).to_client())
    # fl.client.start_client(server_address="pearl.mlkd.tp.vps.inesc-id.pt:8080", client=FlowerClient().to_client())
