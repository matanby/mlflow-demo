#! /usr/bin/env python

import os

import fire
import keras
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
from keras.layers import Dense, Dropout


def _load_dataset():
    # Read the wine-quality csv file.
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wine-quality.csv')
    dataset = pd.read_csv(dataset_path)

    # Split the data into training and test sets. (0.85, 0.15) split.
    train, test = sklearn.model_selection.train_test_split(dataset, test_size=0.15)

    # The predicted column is "quality" which is a scalar from [3, 9]
    x_train = train.drop(["quality"], axis=1)
    y_train = train[["quality"]]
    x_test = test.drop(["quality"], axis=1)
    y_test = test[["quality"]]

    return (x_train, y_train), (x_test, y_test)


def _calc_metrics(actual, pred):
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(actual, pred))
    mae = sklearn.metrics.mean_absolute_error(actual, pred)
    r2 = sklearn.metrics.r2_score(actual, pred)
    return rmse, mae, r2


def _print_metrics(header, **kwargs):
    print('\n' + header)
    for metric_name, metric_value in kwargs.items():
        print('{}: {:.4f}'.format(metric_name, metric_value))

    print()


def train_model(dropout_rate=0.0, batch_size=128, epochs=5, num_units=128, lr=1e-3):
    # load the dataset.
    (x_train, y_train), (x_test, y_test) = _load_dataset()

    # create a simple MLP model.
    _input = keras.Input(x_train.shape[1:])
    x = Dense(units=num_units, activation='relu')(_input)
    x = Dropout(rate=dropout_rate)(x)
    x = Dense(units=num_units, activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x)
    _output = Dense(units=1, activation=None)(x)
    model = keras.Model(_input, _output)

    # compile the model with an MSE loss and an Adam optimizer.
    model.compile(
        loss=keras.losses.mse,
        optimizer=keras.optimizers.Adam(lr),
        metrics=['mse']
    )

    # train the model.
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(x_test, y_test)
    )

    # calculate metrics on the train set.
    train_preds = model.predict(x_train)
    train_rmse, train_mae, train_r2 = _calc_metrics(y_train, train_preds)
    _print_metrics('Train metrics:', RMSE=train_rmse, MAE=train_mae, R2=train_r2)

    # calculate metrics on the test set.
    test_preds = model.predict(x_test)
    test_rmse, test_mae, test_r2 = _calc_metrics(y_test, test_preds)
    _print_metrics('Test metrics:', RMSE=test_rmse, MAE=test_mae, R2=test_r2)


if __name__ == '__main__':
    fire.Fire(train_model)
