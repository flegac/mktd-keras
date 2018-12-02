import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# - 1.1 pandas : Load a dataset from CSV file using pandas library
# - 1.2 numpy : Use numpy to reshape some arrays
# - 1.3 sklearn? : Data normalization (mean and standard deviation)
# - 1.4 Training and validation data separation
# - 1.5 matplot : visualize images / matrix
# - 1.6 matplot : visualize distribution of data by categories


def test_pandas_load_csv():
    expected_shape = (4, 3)

    # load csv file
    data = None
    # data = pd.read_csv('../resources/dataset.csv')

    assert data.shape == expected_shape


def test_reshape_array():
    expected_shape = (4, 3, 5)

    before = np.array(range(4 * 3 * 5))
    after = before

    #  reshape to shape (4,3,5)
    # after = np.reshape(before, expected_shape)

    assert after.shape == expected_shape


def compute_std_dev_and_mean():
    data = np.array([[1, 2, 3, 4, 5, 6],
                     [4, 5, 6, 7, 8, 9]],
                    dtype=np.float)
    data_mean = data.mean(axis=1)
    data_std = data.std(axis=1)
    data_scaled = preprocessing.scale(data, axis=1)

    scaled_mean = data_scaled.mean(axis=1)
    scaled_std = data_scaled.std(axis=1)

    print('data : {} {}'.format(data_mean, data_std))
    print('scaled : {} {}'.format(scaled_mean, scaled_std))


def extract_validation_data():
    X = np.array(range(500)).reshape((100, 5))
    Y = np.array(range(100))
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.10, random_state=None)
    print('{.shape} {.shape} {.shape} {.shape}'.format(x_train,y_train,x_val,y_val))
