import os
import scipy
from flask import Flask, request
from werkzeug.utils import secure_filename

import cv2
from exercices.models import Models
from exercices.visualize import show_confusion_matrix, show_samples
from project.training.training_script import prepare_dataset
import numpy as np


def test_confusion_matrix():
    model = Models.load_model('../resources/model')
    input_shape = model.input_shape[1:]
    train, test = prepare_dataset(input_shape, batch_size=1)

    expected = []
    predictions = []
    errors = []

    for i in range(len(test)):
        x, y = next(test)
        y = np.argmax(y)
        z = np.argmax(model.predict(x))
        expected.append(y)
        predictions.append(z)
        if z != y:
            print('expected: {} actual: {}'.format(y, z))
            errors.append((x, z))

    cm = show_confusion_matrix(expected, predictions)
    print('confusion matrix :')
    print(cm)

    # display errors
    errors = errors[:100]
    show_samples(errors, 10, 10)
