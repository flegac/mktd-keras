from exercices.dataset import *


def test_dataset_loading():
    (x_train, y_train), (x_test, y_test) = Datasets.fashion_mnist()

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) == 60000
    assert len(x_test) == 10000


def test_load_csv():
    data = Datasets.from_csv('../resources/dataset.csv')
    assert data.shape == (4, 3)


def test_reshape_array():
    expected_shape = (4, 3, 5)
    before = Tensors.create(range(4 * 3 * 5))
    after = Tensors.reshape(before, expected_shape)
    assert after.shape == expected_shape


def test_normalization():
    array = np.array([[1, 2, 3, 4, 5, 6],
                      [4, 5, 6, 7, 8, 9]],
                     dtype=np.float)

    data_scaled = DataPreparation.normalize(array)

    mean = array.mean(axis=1)
    std = array.std(axis=1)
    scaled_mean = data_scaled.mean(axis=1)
    scaled_std = data_scaled.std(axis=1)

    print('before : (mean,std) = ({}, {})'.format(mean, std))
    print('after : (mean,std) = ({}, {})'.format(scaled_mean, scaled_std))


def test_dataset_split():
    (_, _), (x, y) = Datasets.mnist()
    (x_train, x_val), (y_train, y_val) = Datasets.split_dataset(x, y)

    assert x_train.shape == (9000, 28, 28)
    assert x_val.shape == (9000,)
    assert y_train.shape == (1000, 28, 28)
    assert y_val.shape == (1000,)
