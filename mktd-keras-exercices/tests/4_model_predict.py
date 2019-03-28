import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from exercices.dataset import Datasets
from exercices.models import Models
from exercices.visualize import show_confusion_matrix, show_image, show_samples

model_path = '../resources/model'


def get_image():
    generator = Datasets.mnist()()
    x, y = next(generator)
    return x


def test_predict_once():
    x = get_image()

    model = Models.load_model(model_path)

    yy = Models.predict(model, x)

    assert yy.shape[1] == 10


def test_confusion_matrix():
    sample_provider = Datasets.mnist()
    x = np.array([_[0] for _ in sample_provider()])
    y = np.array([_[1] for _ in sample_provider()])

    train, validation = Datasets.split_dataset(x, y)
    expected = [np.argmax(_) for _ in validation[1]]

    dataset = ImageDataGenerator(
        rescale=1. / 255
    ).flow(
        validation[0],
        shuffle=False,
        batch_size=1
    )

    model = Models.load_model(model_path)
    yy = model.predict_generator(
        dataset,
        steps=dataset.n
    )

    predictions = [np.argmax(_) for _ in yy]

    cm = show_confusion_matrix(expected, predictions)

    print('confusion matrix :')
    print(cm)

    # display errors
    errors = []
    for i in range(yy.shape[0]):
        if expected[i] != predictions[i]:
            print('expected: {} actual: {}'.format(expected[i], predictions[i]))
            errors.append((x[i], predictions[i]))
    errors = errors[:100]
    show_samples(errors, 10, 10)
