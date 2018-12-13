from flask import Flask, request
from werkzeug.utils import secure_filename

import cv2

app = Flask(__name__)


@app.route("/process", methods=['POST'])
def hello():
    print("processing !")
    file = request.files['input']

    filename = '/tmp/uploads/' + secure_filename(file.filename)

    file.save(filename)

    im = cv2.imread(filename)
    print(type(im))

    return str(im.shape)

app.run()
