import os

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from exercices.dataset import Datasets
from exercices.models import Models

model_path = '../resources/model'


def test_create_model():
    model = Models.create(input_shape=(28, 28, 1), num_classes=10)
    Models.save_model(model, model_path)

    assert os.path.exists(os.path.join(model_path, 'model.h5'))


def test_load_model():
    model = Models.load_model(model_path)
    assert isinstance(model, keras.models.Model)


def test_model_training():
    train, validation = dataset_preparation()

    # TODO : Modify the Model to improve performances !
    model = Models.create(input_shape=(28, 28, 1), num_classes=10)
    Models.print_model_summary(model)
    Models.train(model, train, validation)
    Models.save_model(model, './my_model')

    # evaluate the model
    scores = model.evaluate_generator(
        validation,
        steps=validation.n,
        verbose=0)

    accuracy_percentage = scores[1] * 100
    print("evaluation on unseen dataset : {} = {}".format(model.metrics_names[1], accuracy_percentage))

    # TODO : add batch normalization to Models.create_model() to gain some accuracy
    # https://keras.io/layers/normalization/
    assert accuracy_percentage >= 50, "Bad accuracy ({}%) : {}".format(
        accuracy_percentage,
        "something is wrong :(")


def test_batch_normalization():
    train, validation = dataset_preparation()
    model = Models.load_model('./my_model')

    # evaluate the model
    scores = model.evaluate_generator(
        validation,
        steps=validation.n,
        verbose=0)

    accuracy_percentage = scores[1] * 100
    print("evaluation on unseen dataset : {} = {}".format(model.metrics_names[1], accuracy_percentage))

    # TODO : add batch normalization to Models.create_model() to gain some accuracy
    assert accuracy_percentage >= 65, "Bad accuracy ({}%) : {}".format(
        accuracy_percentage,
        "add a batch normalization layer to your model and retrain it !")


def test_convolutional_network():
    train, validation = dataset_preparation()
    model = Models.load_model('./my_model')

    # evaluate the model
    scores = model.evaluate_generator(
        validation,
        steps=validation.n,
        verbose=0)

    accuracy_percentage = scores[1] * 100
    print("evaluation on unseen dataset : {} = {}".format(model.metrics_names[1], accuracy_percentage))

    # TODO : add convolutional layers to Models.create_model() to improve its accuracy
    # https://keras.io/layers/convolutional/
    assert accuracy_percentage >= 75, "Bad accuracy : ({}%) : {}".format(
        accuracy_percentage,
        "use a Convolutional layer in your model and retrain it !")


def dataset_preparation():
    sample_provider = Datasets.fashion_mnist()
    x = [_[0] for _ in sample_provider()]
    y = [_[1] for _ in sample_provider()]

    # take 1 out of n elements
    # TODO: increase n if the training is taking too long : performance could decrease (or increase, yet it is unlikely)
    n = 20
    x = x[::n]
    y = y[::n]

    train, validation = Datasets.split_dataset(np.array(x), np.array(y))

    # data augmentation : each image is rotated / shifted / scaled
    train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=8,
        width_shift_range=0.08,
        # shear_range=0.3,
        height_shift_range=0.08,
        zoom_range=0.08,
    ).flow(
        train,
        shuffle=True,
        batch_size=32
    )

    # no data augmentation
    validation = ImageDataGenerator(
        rescale=1. / 255
    ).flow(
        validation,
        shuffle=False,
        batch_size=32
    )
    return train, validation
