import os

import keras
import scipy
from flask import Flask, request
from keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
import tensorflow as tf

import cv2
import numpy as np

from exercices.dataset import Tensors
from exercices.models import Models
from exercices.visualize import show_image

app = Flask(__name__)

# Flask / Tensorflow fix
# https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2
global graph
graph = tf.get_default_graph()
print('preload model ...')
model = Models.load_model('../resources/model')
assert isinstance(model, keras.models.Model)
print('model is ready !')


@app.route("/process", methods=['POST'])
def predict():
    print("start processing ...")
    file = request.files['input']
    filename = '/tmp/uploads/' + secure_filename(file.filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print('save uploaded image at : {}'.format(filename))
    file.save(filename)

    print('read image from local disk')
    img = Tensors.from_image(filename)

    print('prepare image for prediction')
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    img = img / 255.

    print('make prediction using model ...')
    with graph.as_default():
        result = Models.predict(model, img)
    print('prediction : {}'.format(result))
    print('---> {}'.format(np.argmax(result)))
    try:
        print('---> {}'.format(np.argmax(result, axis=1)))
    except:
        pass

    return str(np.argmax(result))


app.run()
