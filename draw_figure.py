from datetime import datetime
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_datasets

# Create output folder
time_str = datetime.now().strftime('%m/%d_%H%M%S')
output_dir = f'data/output/{time_str}_figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

colors = [
    ['#EDF4F5', '#C5E3E2', '#9EC6DB']
    # ['#DFE1E2', '#B7DBE3', '#F5E09B'],
    # ['#F4DEBB', '#E1B6B5', '#F9F2C1'],
    # ['#9EE092', '#F8D793', '#D2D2D2'],
    # ['#F9F5F6', '#DDDEDE', '#D882AD']
]

DPI = 150

def draw_prediction():
    regions = CONFIG['regions']
    for region in regions:
        df_llstm = pd.read_excel(f'data/output/06/05_225330_local/predit_data-LSTM.xlsx', index_col='Date', sheet_name=region)
        # df_clstm = pd.read_excel(f'data/output/06/06_133621_centralized/predict_data-C-LSTM.xlsx', sheet_name=region)
        df_flstm_a1 = pd.read_excel(f'data/output/07/02_161544_fl/predit_data-{region}-F-LSTM.xlsx', index_col='Date')
        df_flstm_a2 = pd.read_excel(f'data/output/07/02_161732_fl/predit_data-{region}-F-LSTM.xlsx', index_col='Date')
        df_flstm_a3 = pd.read_excel(f'data/output/07/02_162123_fl/predit_data-{region}-F-LSTM.xlsx', index_col='Date')
        df_flstm_a4 = pd.read_excel(f'data/output/07/02_162305_fl/predit_data-{region}-F-LSTM.xlsx', index_col='Date')
        df_flstm_a5 = pd.read_excel(f'data/output/07/02_162447_fl/predit_data-{region}-F-LSTM.xlsx', index_col='Date')
        df_flstm_a6 = pd.read_excel(f'data/output/07/02_162447_fl/predit_data-{region}-F-LSTM.xlsx', index_col='Date')
        df_flstm_a7 = pd.read_excel(f'data/output/07/02_162636_fl/predit_data-{region}-F-LSTM.xlsx', index_col='Date')

        df = pd.concat([df_llstm,# df_clstm,
                        df_flstm_a1, df_flstm_a2, df_flstm_a3, df_flstm_a4, df_flstm_a5, df_flstm_a6, df_flstm_a7], ignore_index=True)

        # Plot results
        fig, axs = plt.subplots(1, 2, figsize=(24, 6), dpi=DPI)

        dataset = load_datasets(region, ahead=0)
        y_train_origin = dataset.scaler.inverse_transform(
            dataset.y_train_origin.detach().numpy().reshape(-1, 1)).flatten()
        y_test = dataset.scaler.inverse_transform(dataset.y_test.unsqueeze(1).numpy().reshape(-1, 1)).flatten()
        y = pd.Series(np.concatenate((y_train_origin, y_test), axis=0), index=dataset.index)

        colormap = plt.get_cmap('cool')

        axs[0].plot(y, label='Reported cases', c='black')
        for i in range(1, 8):
            axs[0].plot(df_llstm[f'LSTM_ahead{i}'], label=f'L-LSTM ({i} days ahead)')
            # axs[0].plot(y_pred_test, label=f'{MODEL_NAME} (Test)')
            # axs[0].set_title(f'{MODEL_NAME} prediction for {region}')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Number of reported cases')
        axs[0].legend()

        axs[1].plot(y, label='Reported cases', c='black')
        axs[1].plot(df_flstm_a1[f'F-LSTM_ahead1'], label='F-LSTM (1 days ahead)')
        axs[1].plot(df_flstm_a2[f'F-LSTM_ahead2'], label='F-LSTM (2 days ahead)')
        axs[1].plot(df_flstm_a3[f'F-LSTM_ahead3'], label='F-LSTM (3 days ahead)')
        axs[1].plot(df_flstm_a4[f'F-LSTM_ahead4'], label='F-LSTM (4 days ahead)')
        axs[1].plot(df_flstm_a5[f'F-LSTM_ahead5'], label='F-LSTM (5 days ahead)')
        axs[1].plot(df_flstm_a6[f'F-LSTM_ahead6'], label='F-LSTM (6 days ahead)')
        axs[1].plot(df_flstm_a7[f'F-LSTM_ahead7'], label='F-LSTM (7 days ahead)')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Number of reported cases')
        axs[1].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(f'{output_dir}/figure1-prediction_{region}.png')
        plt.cla()
        plt.close()


def draw_performance():
    df_llstm = pd.read_excel(f'data/output/06/05_225330_local/performance.xlsx')
    df_clstm = pd.read_excel(f'data/output/07/02_165413_centralized/performance.xlsx')
    # Only use the last round performance
    df_flstm_a1 = pd.read_excel(f'data/output/07/02_160904_fl_ahead1/performance.xlsx').iloc[[-1]]
    df_flstm_a2 = pd.read_excel(f'data/output/07/02_161544_fl_ahead2/performance.xlsx').iloc[[-1]]
    df_flstm_a3 = pd.read_excel(f'data/output/07/02_161732_fl_ahead3/performance.xlsx').iloc[[-1]]
    df_flstm_a4 = pd.read_excel(f'data/output/07/02_162123_fl_ahead4/performance.xlsx').iloc[[-1]]
    df_flstm_a5 = pd.read_excel(f'data/output/07/02_162305_fl_ahead5/performance.xlsx').iloc[[-1]]
    df_flstm_a6 = pd.read_excel(f'data/output/07/02_162447_fl_ahead6/performance.xlsx').iloc[[-1]]
    df_flstm_a7 = pd.read_excel(f'data/output/07/02_162636_fl_ahead7/performance.xlsx').iloc[[-1]]

    df = pd.concat([df_llstm, df_clstm, df_flstm_a1, df_flstm_a2, df_flstm_a3, df_flstm_a4, df_flstm_a5,
                    df_flstm_a6, df_flstm_a7], ignore_index=True)
    print(df)

    df = df.rename(columns={'RMSE_normalized': 'nRMSE', 'MAE_normalized': 'nMAE'})

    # Min-Max normalization for SMAPE
    min_value = df['SMAPE'].min()
    max_value = df['SMAPE'].max()
    df['nSMAPE'] = (df['SMAPE'] - min_value) / (max_value - min_value)

    df = df.groupby(['Ahead', 'Model']).agg({
        'nRMSE': 'mean', 'nMAE': 'mean', 'nSMAPE': 'mean', 'R2': 'mean'
    }).reset_index()
    print(df)

    df = df.groupby('Ahead')
    print(df)

    # Draw bar chart for each group
    for key, group in df:
        print(f'key={key}, group={group}')
        group = group.sort_values(by='Model', ascending=False)

        for i in range(len(colors)):
            plt.figure(num=None, figsize=(8, 8), dpi=DPI)
            ax = group.plot(kind='bar', x='Model', y=['nRMSE', 'nMAE', 'nSMAPE'], color=colors[i],
                       edgecolor='white', width=0.9)

            # Draw value above each bar
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', label_type='edge')
            # plt.xlabel('Model')
            # plt.ylabel('Values')
            # plt.title(f'key = {key}')
            plt.legend(loc='best')
            # plt.show()
            plt.xticks(rotation=0)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.savefig(f'{output_dir}/figure2-ahead{key}_color{i}.png')
            plt.cla()
            plt.close()


def draw_rounds():
    regions = CONFIG['regions']
    df_flstm_a1 = pd.read_excel(f'data/output/07/02_160338_fl/performance.xlsx')
    df = df_flstm_a1.rename(columns={'RMSE_normalized': 'nRMSE', 'MAE_normalized': 'nMAE'})

    for region in regions:
        data = df[df['Region'] == region]
        for indice in ['nRMSE', 'nMAE', 'nSMAPE']:
            plt.figure(num=None, figsize=(6, 6))
            plt.plot([x for x in range(1, len(data[indice]) + 1)], data[indice], label=indice)
            plt.xlabel('Rounds')
            plt.ylabel(indice)
            plt.legend()
            plt.tight_layout()
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.savefig(f'{output_dir}/figure3_{region}.png')
            plt.cla()
            plt.close()


if __name__ == '__main__':
    # draw_prediction()
    draw_performance()
    # draw_rounds()
