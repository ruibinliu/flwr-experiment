class Dataset:
    def __init__(self, index, x_train, x_test, y_train, y_test, scaler, X_train_origin, y_train_origin):
        self.index = index
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        self.X_train_origin = X_train_origin
        self.y_train_origin = y_train_origin
