from datetime import datetime
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG

# Create output folder
time_str = datetime.now().strftime('%m/%d_%H%M%S')
output_dir = f'data/output/{time_str}_figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# df_mo = pd.read_excel(f'data/output/04/26_155817_local/performance_Macau.xlsx', sheet_name='Macau')
# df_pt = pd.read_excel(f'data/output/04/26_155817_local/performance_Macau.xlsx', sheet_name='Portugal')
# df_gd = pd.read_excel(f'data/output/04/26_155817_local/performance_Macau.xlsx', sheet_name='Guangdong')
# scaler 0, 1, epoch=1
df_pt = pd.read_excel(f'data/output/05/09_110222_fl/performance_Portugal.xlsx', sheet_name='Sheet')
df_gd = pd.read_excel(f'data/output/05/09_110222_fl/performance_Guangdong.xlsx', sheet_name='Sheet')
df_mo = pd.read_excel(f'data/output/05/09_110222_fl/performance_Macau.xlsx', sheet_name='Sheet')
# scaler 0, 1, epoch=2
# df_pt = pd.read_excel(f'data/output/05/09_111638_fl/performance_Portugal.xlsx', sheet_name='Sheet')
# df_gd = pd.read_excel(f'data/output/05/09_111638_fl/performance_Guangdong.xlsx', sheet_name='Sheet')
# df_mo = pd.read_excel(f'data/output/05/09_111638_fl/performance_Macau.xlsx', sheet_name='Sheet')
# scaler 0, 1, epoch=3
# df_pt = pd.read_excel(f'data/output/05/09_112617_fl/performance_Portugal.xlsx', sheet_name='Sheet')
# df_gd = pd.read_excel(f'data/output/05/09_112617_fl/performance_Guangdong.xlsx', sheet_name='Sheet')
# df_mo = pd.read_excel(f'data/output/05/09_112717_fl/performance_Macau.xlsx', sheet_name='Sheet')
# scaler 0, 1, epoch=4
# df_pt = pd.read_excel(f'data/output/05/09_113207_fl/performance_Portugal.xlsx', sheet_name='Sheet')
# df_gd = pd.read_excel(f'data/output/05/09_113207_fl/performance_Guangdong.xlsx', sheet_name='Sheet')
# df_mo = pd.read_excel(f'data/output/05/09_113207_fl/performance_Macau.xlsx', sheet_name='Sheet')
# scaler 0, 1, epoch=5
# df_pt = pd.read_excel(f'data/output/05/09_115914_fl/performance_Portugal.xlsx', sheet_name='Sheet')
# df_gd = pd.read_excel(f'data/output/05/09_115914_fl/performance_Guangdong.xlsx', sheet_name='Sheet')
# df_mo = pd.read_excel(f'data/output/05/09_115914_fl/performance_Macau.xlsx', sheet_name='Sheet')
# scaler 0, 1000
# df_mo = pd.read_excel(f'data/output/05/09_151256_fl/performance_Portugal.xlsx', sheet_name='Sheet')
# df_pt = pd.read_excel(f'data/output/05/09_151301_fl/performance_Guangdong.xlsx', sheet_name='Sheet')
# df_gd = pd.read_excel(f'data/output/05/09_151306_fl/performance_Macau.xlsx', sheet_name='Sheet')
# L-LSTM, C-LSTM, F-LSTM
# df_mo = pd.read_excel(f'data/output/05/09_153651_local/performance_Macau.xlsx', sheet_name='Macau')
# df_pt = pd.read_excel(f'data/output/05/09_153651_local/performance_Macau.xlsx', sheet_name='Portugal')
# df_gd = pd.read_excel(f'data/output/05/09_153651_local/performance_Macau.xlsx', sheet_name='Guangdong')


def draw_performance():
    plt.figure(num=None, figsize=(12, 12))
    indices = ['RMSE', 'MAE', 'SMAPE', 'R2']
    for i in range(len(indices)):
        plt.subplot(2, 2, i + 1)
        indice = indices[i]
        regions = ['Guangdong', 'Macau', 'Portugal']
        x = np.arange(len(regions))
        width = 0.2
        x_llstm = x
        x_clstm = x + width
        x_flstm = x + 2 * width

        llstm = [df_gd.loc[df_gd['Model'] == 'L-LSTM'][indice].iloc[0],
                 df_mo.loc[df_mo['Model'] == 'L-LSTM'][indice].iloc[0],
                 df_pt.loc[df_pt['Model'] == 'L-LSTM'][indice].iloc[0]]
        clstm = [df_gd.loc[df_gd['Model'] == 'C-LSTM'][indice].iloc[0],
                 df_mo.loc[df_mo['Model'] == 'C-LSTM'][indice].iloc[0],
                 df_pt.loc[df_pt['Model'] == 'C-LSTM'][indice].iloc[0]]
        flstm = []

        best_round_index = []
        for df in [df_gd, df_mo, df_pt]:
            best_index = 0
            best = df.loc[df['Model'].str.contains('FL_LSTM')][indice].iloc[0]
            mask1 = len(df.loc[df['Model'].str.contains('FL_LSTM')][indice])
            mask2 = df.loc[df['Model'].str.contains('FL_LSTM')][indice]
            for i in range(len(df.loc[df['Model'].str.contains('FL_LSTM')][indice])):
                if indice == 'R2':
                    # The larger, the better
                    mask = df.loc[df['Model'].str.contains('FL_LSTM')][indice].iloc[i]
                    if df.loc[df['Model'].str.contains('FL_LSTM')][indice].iloc[i] > best:
                        best_index = i
                        best = df.loc[df['Model'].str.contains('FL_LSTM')][indice].iloc[i]
                else:
                    # The smaller, the better
                    mask = df.loc[df['Model'].str.contains('FL_LSTM')][indice].iloc[i]
                    if df.loc[df['Model'].str.contains('FL_LSTM')][indice].iloc[i] < best:
                        best_index = i
                        best = df.loc[df['Model'].str.contains('FL_LSTM')][indice].iloc[i]

            flstm.append(best)
            best_round_index.append(best_index)

        plt.bar(x_llstm, llstm, width=width, label='L-LSTM')
        plt.bar(x_clstm, clstm, width=width, label='C-LSTM')
        plt.bar(x_flstm, flstm, width=width, label='F-LSTM')
        # Replace x axis labels
        plt.xticks(x + width, labels=regions)

        ylabel = 'R square' if indice == 'R2' else indice
        title = ylabel + '\n' + f'FL-LSTM: Best {ylabel} of [GD, MO, PT] is round {best_round_index}.'

        plt.title(title)
        plt.xlabel('Region')
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance.png')
    plt.cla()
    plt.close()


def draw_rounds():
    plt.figure(num=None, figsize=(12, 12))
    indices = ['RMSE', 'MAE', 'SMAPE', 'R2']
    for i in range(len(indices)):
        plt.subplot(2, 2, i + 1)
        indice = indices[i]

        plt.plot(df_gd.loc[df_gd['Model'].str.contains('FL_LSTM')][indice], label='Guangdong')
        plt.plot(df_mo.loc[df_mo['Model'].str.contains('FL_LSTM')][indice], label='Macau')
        plt.plot(df_pt.loc[df_pt['Model'].str.contains('FL_LSTM')][indice], label='Portugal')

        plt.title('R^2' if indice == 'R2' else indice)
        plt.xlabel('Rounds')
        plt.ylabel('R^2' if indice == 'R2' else indice)
        plt.grid()
        plt.legend()
    plt.suptitle(f'Federated learning LSTM (epoch=1)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance-rounds.png')
    plt.cla()
    plt.close()

    with open(f'{output_dir}/config.txt', 'w') as file:
        for key, value in CONFIG.items():
            file.write(f'{key}={value}\n')


if __name__ == '__main__':
    # draw_performance()
    draw_rounds()
