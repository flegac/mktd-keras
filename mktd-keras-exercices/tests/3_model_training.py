import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from exercices.dataset import Datasets
from exercices.models import Models


def test_model_training():
    train, validation = dataset_preparation()

    # TODO : Modify the Model to improve performances !
    model_path = '../resources/model'
    try:
        model = Models.load_model(model_path)
    except:
        model = Models.create(input_shape=(28, 28, 1), num_classes=10)
        Models.train(model, train, validation)
        Models.save_model(model, model_path)

    # evaluate the model
    scores = model.evaluate_generator(
        validation,
        steps=validation.n,
        verbose=0)

    accuracy_percentage = scores[1] * 100
    print("evaluation on unseen dataset : {} = {}".format(model.metrics_names[1], accuracy_percentage))

    # TODO : add batch normalization to Models.create_model() and see real improvements
    # https://keras.io/layers/normalization/
    assert accuracy_percentage >= 70, "Bad accuracy ({}%) : {}".format(
        accuracy_percentage,
        "add a batch normalization layer to your model !")

    # TODO : add convolutional layers to to Models.create_model() and boost the efficiency of the model
    # https://keras.io/layers/convolutional/
    assert accuracy_percentage >= 90, "Bad accuracy : ({}%) : {}".format(
        accuracy_percentage,
        "use a Convolutional layer !")

    # TODO : try to get the best score, learn from state of the art networks !
    # https://keras.io/applications/
    assert accuracy_percentage >= 90, "Bad accuracy : ({}%) : {}".format(
        accuracy_percentage,
        "Loonk at states of the art networks : https://keras.io/applications/")


def dataset_preparation():
    sample_provider = Datasets.mnist()
    x = np.array([_[0] for _ in sample_provider()])
    y = np.array([_[1] for _ in sample_provider()])

    train, validation = Datasets.split_dataset(x, y)

    # data augmentation : each image is rotated / shifted / scaled
    train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=8,
        width_shift_range=0.08,
        shear_range=0.3,
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
