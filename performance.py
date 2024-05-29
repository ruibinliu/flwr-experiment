import numpy as np
from openpyxl import load_workbook, Workbook


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
                     index_of_test_begin, index_of_test_end, ahead=0):
    # Load existing Excel file or create a new one if it doesn't exist
    file_path = f'{output_dir}/performance_{region}.xlsx'
    try:
        wb = load_workbook(file_path)
        ws = wb.active
    except FileNotFoundError:
        wb = Workbook()
        ws = wb.active
        ws.append(['Region', 'RMSE', 'RMSE_normalized',
                   'MAE', 'MAE_normalized',
                   'SMAPE', 'R2', 'Model',
                   'Train_start', 'Train_end', 'Test_start', 'Test_end', 'Ahead'])
    ws.append([region,
               rmse, rmse_normalized,
               mae, mae_normalized,
               smape, r2, model,
               index_of_train_begin, index_of_train_end,
               index_of_test_begin, index_of_test_end, ahead + 1])
    wb.save(file_path)
