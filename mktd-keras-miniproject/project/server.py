import os

from flask import Flask, request
from werkzeug.utils import secure_filename

import cv2

from project.predict.predict_lib import init_model

app = Flask(__name__)


@app.route("/process", methods=['POST'])
def predict():
    print("processing !")
    file = request.files['input']

    filename = '/tmp/uploads/' + secure_filename(file.filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print('save file at : {}'.format(filename))
    file.save(filename)
    im = cv2.imread(filename)
    print(type(im))

    print('load model')
    model = init_model('../resources/model')

    print('predict with model ...')
    result = predict(model, im)

    return str(result)


app.run()
