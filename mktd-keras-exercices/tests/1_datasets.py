from functools import reduce

from exercices.dataset import *


def test_dataset_loading():
    # TODO: implements Datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = Datasets.fashion_mnist()

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) == 60000
    assert len(x_test) == 10000


def test_load_csv():
    # TODO: implements Datasets.from_csv
    data = Datasets.from_csv('../resources/dataset.csv')

    assert data.shape == (4, 3)


def test_reshape_array():
    expected_shape = (4, 3, 5)
    size = reduce(lambda x, y: x * y, expected_shape)
    before = Tensors.create(range(size))

    # TODO: implements Tensors.reshape
    after = Tensors.reshape(before, expected_shape)

    assert after.shape == expected_shape


def test_normalization():
    array = np.array([[1, 2, 3],
                      [3, 4, 5]],
                     dtype=np.float)
    before = (array, array.mean(axis=1), array.std(axis=1))

    # TODO: implements DataPreparation.normalize
    array_scaled = DataPreparation.normalize(array)

    mean = array_scaled.mean(axis=1)
    std = array_scaled.std(axis=1)

    print('before : (array, mean, std) = ({}, {})'.format(*before))
    print('after : (array, mean, std) = ({}, {})'.format(array_scaled, mean, std))

    assert np.allclose(mean, [0., 0.])
    assert np.allclose(std, [1., 1.])


def test_dataset_split():
    (_, _), (x, y) = Datasets.mnist()
    (x_train, x_val), (y_train, y_val) = Datasets.split_dataset(x, y)

    assert x_train.shape == (9000, 28, 28)
    assert x_val.shape == (9000,)
    assert y_train.shape == (1000, 28, 28)
    assert y_val.shape == (1000,)
