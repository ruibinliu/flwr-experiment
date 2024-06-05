import os.path

import numpy as np
import pandas as pd
import portalocker


def smape(y_true, y_pred):
    """
    Compute SMAPE（Symmetric Mean Absolute Percentage Error）
    """
    try:
        n = len(y_true)
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred))  # / 2
        smape = np.sum(numerator / denominator) * (100.0 / n)
        return smape
    except:
        return -1


def save_performance(output_dir, region, rmse, rmse_normalized,
                     mae, mae_normalized,
                     smape, r2, model, index_of_dataset_begin, index_of_dataset_end,
                     index_of_train_begin, index_of_train_end,
                     index_of_test_begin, index_of_test_end, ahead=0, nround=-1):
    new_row = {
        'Region': region,
        'RMSE': rmse,
        'RMSE_normalized': rmse_normalized,
        'MAE': mae,
        'MAE_normalized': mae_normalized,
        'SMAPE': smape,
        'R2': r2,
        'Model': model,
        'Train_start': index_of_train_begin,
        'Train_end': index_of_train_end,
        'Test_start': index_of_test_begin,
        'Test_end': index_of_test_end,
        'Ahead': ahead + 1,
        'FL_Round': nround
    }

    # Load existing Excel file or create a new one if it doesn't exist
    output_file = f'{output_dir}/performance.xlsx'

    if not os.path.exists(output_file):
        with open(output_file, 'wb') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            df = pd.DataFrame(columns=new_row.keys())
            with pd.ExcelWriter(f, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            portalocker.unlock(f)  # 解锁文件

    with open(output_file, 'r+b') as f:
        portalocker.lock(f, portalocker.LOCK_EX)  # 排他锁定文件

        try:
            df = pd.read_excel(f, engine='openpyxl')
        except ValueError:
            df = pd.DataFrame()

        f.seek(0)

        df = pd.concat([df, pd.DataFrame(new_row, index=[0])])

        f.truncate()
        with pd.ExcelWriter(f, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

        portalocker.unlock(f)  # 解锁文件
