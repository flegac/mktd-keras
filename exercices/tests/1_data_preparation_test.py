import pandas as pd
import numpy as np
from sklearn import preprocessing


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
    data = np.array([[1, 3, 5, 2, 1, 1, 4, 3, 7, 9, 12, 3, 2, 1, 3],
                     [1, 3, 5, 2, 1, 1, 4, 3, 7, 9, 12, 3, 2, 1, 3]],
                    dtype=np.float)
    data_scaled = preprocessing.scale(data)

    data_scaled.mean()
    data_scaled.std()
