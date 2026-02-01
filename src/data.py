import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def prepare_datasets(val_size=5000):
    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_val = x_train_full[:val_size]
    y_val = y_train_full[:val_size]

    x_train = x_train_full[val_size:]
    y_train = y_train_full[val_size:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)