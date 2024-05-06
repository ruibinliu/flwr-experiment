CONFIG = {
    # Load data
    'window_size': 7,
    'input_cols': ['NewCases'],
    'output_col': ['NewCases'],

    # LSTM Hyper Parameters
    'num_epochs': 1,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0,
    'learning_rate': 0.01,

    # Early Stopping
    'patience': 10,  # Early Stopping patience

    # Others
    'regions': ['Portugal', 'Guangdong', 'Macau'],
}
