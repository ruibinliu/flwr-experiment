import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

output_dir = f'data/output/04/26_155817_local/figures/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_mo = pd.read_excel(f'data/output/04/26_155817_local/performance_Macau.xlsx', sheet_name='Macau')
df_pt = pd.read_excel(f'data/output/04/26_155817_local/performance_Macau.xlsx', sheet_name='Portugal')
df_gd = pd.read_excel(f'data/output/04/26_155817_local/performance_Macau.xlsx', sheet_name='Guangdong')


def draw_performance():
    plt.figure(num=None, figsize=(12, 12))
    indices = ['RMSE', 'MAE', 'SMAPE', 'R^2']
    for i in range(len(indices)):
        plt.subplot(2, 2, i + 1)
        indice = indices[i]
        regions = ['Guangdong', 'Macau', 'Portugal']
        x = np.arange(len(regions))
        width = 0.2
        x_gd = x
        x_mo = x + width
        x_pt = x + 2 * width

        llstm = [df_gd.loc[df_gd['Model'] == 'L-LSTM'][indice].iloc[0],
                 df_mo.loc[df_mo['Model'] == 'L-LSTM'][indice].iloc[0],
                 df_pt.loc[df_pt['Model'] == 'L-LSTM'][indice].iloc[0]]
        clstm = [df_gd.loc[df_gd['Model'] == 'C-LSTM'][indice].iloc[0],
                 df_mo.loc[df_mo['Model'] == 'C-LSTM'][indice].iloc[0],
                 df_pt.loc[df_pt['Model'] == 'C-LSTM'][indice].iloc[0]]
        flstm = [df_gd.loc[df_gd['Model'].str.contains('FL-LSTM')][indice].iloc[0],
                 df_mo.loc[df_mo['Model'].str.contains('FL-LSTM')][indice].iloc[0],
                 df_pt.loc[df_pt['Model'].str.contains('FL-LSTM')][indice].iloc[0]]

        plt.bar(x_gd, llstm, width=width, label='L-LSTM')
        plt.bar(x_mo, clstm, width=width, label='C-LSTM')
        plt.bar(x_pt, flstm, width=width, label='F-LSTM')
        # Replace x axis labels
        plt.xticks(x + width, labels=regions)

        plt.title(indice)
        plt.xlabel('Region')
        plt.ylabel(indice)
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance.png')
    plt.cla()
    plt.close()


def draw_rounds():
    plt.figure(num=None, figsize=(12, 12))
    indices = ['RMSE', 'MAE', 'SMAPE', 'R^2']
    for i in range(len(indices)):
        plt.subplot(2, 2, i + 1)
        indice = indices[i]

        plt.plot(df_gd.loc[df_pt['Model'].str.contains('FL-LSTM')][indice], label='Guangdong')
        plt.plot(df_mo.loc[df_pt['Model'].str.contains('FL-LSTM')][indice], label='Macau')
        plt.plot(df_pt.loc[df_pt['Model'].str.contains('FL-LSTM')][indice], label='Portugal')

        plt.title(indice)
        plt.xlabel('Rounds')
        plt.ylabel(indice)
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance-rounds.png')
    plt.cla()
    plt.close()


if __name__ == '__main__':
    draw_performance()
    draw_rounds()
