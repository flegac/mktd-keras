import pandas as pd
import numpy as np


# - 1.1 pandas : Load a dataset from CSV file using pandas library
# - 1.2 numpy : Use numpy to reshape some arrays
# - 1.3 sklearn? : Data normalization (mean and standard deviation)
# - 1.4 Training and validation data separation
# - 1.5 matplot : visualize images / matrix
# - 1.6 matplot : visualize distribution of data by categories


def test_pandas_load_csv():
    # load csv file
    data = pd.read_csv('../resources/dataset.csv')
    expected = (4, 3)

    assert data.shape == expected


def test_reshape_array():
    expected = (4, 3, 5)

    before = np.array(range(4 * 3 * 5))

    #  reshape to shape (4,3,5)
    after = None

    assert after.shape == expected
