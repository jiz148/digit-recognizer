"""
utilities
"""
import os
import numpy as np

PWD = os.path.dirname(os.path.realpath(__file__))


def load_datas():
    """
    load datas from os
    40,000 pictures as train data, 2000 as test data

    :return:
    """
    data = load_dataset()
    data_modified = data[1:, :]
    train_set_x_orig = data_modified[1:39999, 1:]
    train_set_y_orig = data_modified[1:39999, 0]
    test_set_x_orig = data_modified[40000:, 1:]
    test_set_y_orig = data_modified[40000:, 0]
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def load_dataset():
    """
    load data from os

    :return: train data
    """
    train_data_csv = os.path.join(PWD, 'datasets', 'train.csv')
    train_data = np.genfromtxt(train_data_csv, delimiter=',')
    return train_data




