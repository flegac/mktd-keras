import os
import scipy
from flask import Flask, request
from werkzeug.utils import secure_filename
import tensorflow as tf

import cv2
import numpy as np

from exercices.models import Models

app = Flask(__name__)

# Flask / Tensorflow fix
# https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2
global graph
graph = tf.get_default_graph()
print('preload model ...')
model = Models.load_model('../resources/model')
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
    img = cv2.imread(filename)
    img = img / 255.
    img = scipy.resize(img, (128, 128, 3))
    print(type(img))
    print(img.shape)

    print('make prediction using model ...')
    with graph.as_default():
        result = Models.predict(model, img)
    print('prediction : {}'.format(result))

    return str(np.argmax(result))


app.run()
