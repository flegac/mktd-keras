import cv2

import pandas
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split


class Tensors:
    """
    A Tensor is just a multi-dimensional array of numbers.
    It has two main attributes:
     - a shape : an integer array giving the size of each dimension,
     - a type : in memory representation of values (float8, float16, int8, ...)

    Documentation:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
        https://www.tensorflow.org/guide/tensors
    """

    @staticmethod
    def create(array):
        return np.array(array)

    @staticmethod
    def from_image(path: str):
        return cv2.imread(path)

    @staticmethod
    def reshape(array: np.ndarray, shape: tuple):
        # TODO : use numpy reshape function
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

        return array.reshape(shape)
        # raise NotImplementedError()


class Datasets:
    """
    Datasets in machine learning are collections of Samples.
    A Sample is a couple (x,y) of two Tensors.
    The learning is done using such samples :
        - x is the input data (ex: the picture of a dog)
        - y is the expected output given x (we call it ground truth : TR)

    Most Keras Datasets comes as two distinct collections of Samples :
        - train : collection of samples (x_train, y_train) used for training
        - test : collection of samples (x_test, y_test) used for validation after training

    Documentation:
        https://keras.io/datasets/
    """

    @staticmethod
    def mnist():
        # if keras can not download mnist ...
        # copy: https://s3.amazonaws.com/img-datasets/mnist.npz
        # to: ~/.keras/datasets/

        train, test = mnist.load_data()
        # concatenate train & test
        x = np.stack([*train[0], *test[0]])
        y = np.stack([*train[1], *test[1]])

        # reshape
        x = x.reshape(x.shape[0], 28, 28, 1)
        y = to_categorical(y)

        def dataset_generator():
            for i in range(x.shape[0]):
                yield x[i], y[i]

        return dataset_generator

    @staticmethod
    def fashion_mnist():
        # TODO : get the fashion MNIST dataset from keras and create a generator from it
        # https://keras.io/datasets/
        return fashion_mnist.load_data()

        # raise NotImplementedError()

    @staticmethod
    def from_csv(path: str):
        # TODO : use pandas to read a csv file
        # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

        return pandas.read_csv(path)
        # raise NotImplementedError()

    @staticmethod
    def split_dataset(x: np.ndarray, y: np.ndarray, test_size=0.1):
        assert x.shape[0] == y.shape[0]
        # TODO : use sklearn.train_test_split to split a dataset in two datasets (training and validation)
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=None)
        return (x_train, y_train), (x_test, y_test)
        # raise NotImplementedError()


class DataPreparation:
    """
    Data preparation is maybe the most time consuming operation in the process of machine learning.
    Raw data usually comes with flaws making it hard to learn from :
        - noise : data acquisition is not perfect,
        - uneven distribution : some features are over or under represented,
        - missing data : some part of the data is missing for some Samples ...

    Data preparation is the process of making the data ready to be learned from.
        - removing or fixing incomplete data,
        - mapping words to vectors,
        - normalizing vectors ...

    Documentation:
        https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    """

    @staticmethod
    def normalize(array: np.ndarray):
        # TODO : rescale numpy array to feat normal distribution
        # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

        return preprocessing.scale(array, axis=1)
        # raise NotImplementedError()
