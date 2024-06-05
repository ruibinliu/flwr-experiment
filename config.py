CONFIG = {
    # Load data
    'data_path': 'data/input/covid19_data.xlsx',
    'moving_average': 7,  # Smooth the original data
    'input_cols': ['NewCases'],
    'output_col': ['NewCases'],
    'input_len': 7,  # Use $input_len previous data as input.
    'output_len': 7,  # Predict the next 7 days

    # LSTM Hyper Parameters
    'num_epochs': 1000,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0,
    'learning_rate': 0.01,

    # Early Stopping
    'patience': 10,  # Early Stopping patience

    # Others
    'regions': ['Portugal', 'Guangdong', 'Macau'],
}
