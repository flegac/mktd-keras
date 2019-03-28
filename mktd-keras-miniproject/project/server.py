import os

from flask import Flask, request
import tensorflow as tf
from werkzeug.utils import secure_filename

from exercices.models import Models

app = Flask(__name__)


# Flask / Tensorflow fix
# https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2
# global graph
# graph = tf.get_default_graph()
# print('preload model ...')
# model = Models.load_model('../resources/model')
# assert isinstance(model, keras.models.Model)
# print('model is ready !')


@app.route("/process", methods=['POST'])
def predict():
    print("start processing ...")
    file = request.files['input']
    filename = '/tmp/uploads/' + secure_filename(file.filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    result = -1  # category index (0-9)

    return str(result)


app.run()
